import os


import copy
import pdb
from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn

from .utils import get_class_ids_per_task, get_class_names

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec





class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.visual = model.visual
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.token_embedding = model.token_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        # pdb.set_trace()
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.adapter = nn.Linear(512, 512, bias=False ,device=device)
        self.clip_type = model.dtype


        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias



    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text).type(self.clip_type)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_image(self, image):
         # 确保输入数据类型与 self.visual 的权重类型一致
        image = image.to(self.clip_type)
        return self.visual(image)

    
    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        image = image.type(torch.float16)
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)


        with torch.no_grad():
            image_features = self.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.type(self.dtype).detach()).type(self.clip_type)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)
        
        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def adaptation(self, task_id, threshold=0):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id>0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-len(self.class_ids_per_task[task_id])]):
                diff = torch.cdist(self.class_name_features[-len(self.class_ids_per_task[task_id]):].type(torch.float32), class_name_feature.unsqueeze(0).type(torch.float32)).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_new_class = torch.unique(indices[:,1]) + self.cfg.initial_increment+(task_id-1) * self.cfg.increment
            num_hard_class = self.hard_new_class.shape[0]
            self.hard_pairs = indices
            self.hard_pairs[:,1] = self.hard_pairs[:,1]+self.cfg.initial_increment+(task_id-1) * self.cfg.increment
    def get_old_edge_samples(self, batch_size):
        random_select = torch.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], self.old_edge_samples_nearest_labels[random_select]


    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4* torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean()-max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            dist = (weight_new - weight_old).abs()
            U_old, S_old, V_old = torch.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = (P_new - torch.diag(S_old)@V_old).abs()
            mask = dist / dist.max()
            mask += self.mix_b
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + torch.diag(S_old)@V_old * (1-mask)
            weight = U_old @ right
            self.adapter.weight.data = weight
            return




class DINOv2Encoder(nn.Module):
    """DINOv2 ViT-B/14 visual encoder (frozen). Loads from pretrained checkpoint."""
    def __init__(self, weights_path=None, device="cuda"):
        super().__init__()
        self.embed_dim = 768
        
        # Build a minimal DINOv2 ViT-B/14 
        # patch_size=14, embed_dim=768, depth=12, num_heads=12
        from torchvision.models.vision_transformer import VisionTransformer
        self.vit = VisionTransformer(
            image_size=224, patch_size=14, 
            num_layers=12, num_heads=12, 
            hidden_dim=768, mlp_dim=768*4,
            num_classes=0,  # no classification head
        )
        
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            # Map DINOv2 keys to torchvision ViT keys
            new_sd = {}
            for k, v in state_dict.items():
                # patch_embed
                if k == "patch_embed.proj.weight":
                    new_sd["conv_proj.weight"] = v
                elif k == "patch_embed.proj.bias":
                    new_sd["conv_proj.bias"] = v
                elif k == "cls_token":
                    new_sd["class_token"] = v
                elif k == "pos_embed":
                    # DINOv2 pos_embed: [1, 1370, 768] -> interpolate to [1, 257, 768]
                    cls_pe = v[:, :1]
                    patch_pe = v[:, 1:]
                    # 37x37 -> 16x16
                    old_size = int(patch_pe.shape[1] ** 0.5)  # 37
                    patch_pe = patch_pe.reshape(1, old_size, old_size, 768).permute(0, 3, 1, 2)
                    patch_pe = nn.functional.interpolate(patch_pe, size=(16, 16), mode="bicubic", align_corners=False)
                    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, 256, 768)
                    new_sd["encoder.pos_embedding"] = torch.cat([cls_pe, patch_pe], dim=1)
                elif k.startswith("blocks."):
                    # blocks.0.attn.qkv.weight -> encoder.layers.encoder_layer_0.self_attention...
                    parts = k.split(".")
                    layer_idx = parts[1]
                    rest = ".".join(parts[2:])
                    prefix = f"encoder.layers.encoder_layer_{layer_idx}"
                    
                    if rest == "norm1.weight":
                        new_sd[f"{prefix}.ln_1.weight"] = v
                    elif rest == "norm1.bias":
                        new_sd[f"{prefix}.ln_1.bias"] = v
                    elif rest == "norm2.weight":
                        new_sd[f"{prefix}.ln_2.weight"] = v
                    elif rest == "norm2.bias":
                        new_sd[f"{prefix}.ln_2.bias"] = v
                    elif rest == "attn.qkv.weight":
                        new_sd[f"{prefix}.self_attention.in_proj_weight"] = v
                    elif rest == "attn.qkv.bias":
                        new_sd[f"{prefix}.self_attention.in_proj_bias"] = v
                    elif rest == "attn.proj.weight":
                        new_sd[f"{prefix}.self_attention.out_proj.weight"] = v
                    elif rest == "attn.proj.bias":
                        new_sd[f"{prefix}.self_attention.out_proj.bias"] = v
                    elif rest == "mlp.fc1.weight":
                        new_sd[f"{prefix}.mlp.0.weight"] = v
                    elif rest == "mlp.fc1.bias":
                        new_sd[f"{prefix}.mlp.0.bias"] = v
                    elif rest == "mlp.fc2.weight":
                        new_sd[f"{prefix}.mlp.3.weight"] = v
                    elif rest == "mlp.fc2.bias":
                        new_sd[f"{prefix}.mlp.3.bias"] = v
                elif k == "norm.weight":
                    new_sd["encoder.ln.weight"] = v
                elif k == "norm.bias":
                    new_sd["encoder.ln.bias"] = v
            
            missing, unexpected = self.vit.load_state_dict(new_sd, strict=False)
            loaded = len(new_sd) - len(unexpected)
            print(f"Loaded DINOv2: {loaded} params mapped, {len(missing)} missing, {len(unexpected)} unexpected")
        
        # Freeze all parameters
        for p in self.vit.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Extract features from encoder, bypassing classification head
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]  # CLS token, shape [B, 768]


class ClassIncrementalDINO(nn.Module):
    """CLIP text encoder + DINOv2 visual encoder + trainable projection adapter."""
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        
        # CLIP for text encoding only
        clip_model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        self.clip_type = clip_model.dtype
        
        # Both CLIP visual (512d) and DINOv2 (768d) - concatenated features
        self.visual = clip_model.visual  # CLIP visual encoder (frozen)
        dino_weights = getattr(cfg, 'dino_weights', '/mnt/datasets/dinov2_vitb14.pth')
        self.use_dino = os.path.exists(dino_weights)
        if self.use_dino:
            self.dino = DINOv2Encoder(dino_weights, device).to(device)
            visual_dim = 512 + 768  # CLIP + DINOv2 concatenated
            print(f"Using CLIP+DINOv2 fusion (dim={visual_dim})")
        else:
            visual_dim = 512
            print(f"DINOv2 not found, using CLIP only (dim={visual_dim})")
        
        # Projection: concat_dim -> 512 (CLIP text space)
        self.adapter = nn.Linear(visual_dim, 512, bias=False, device=device)
        
        # Override transforms for DINOv2 (224x224, ImageNet normalization)
        from torchvision import transforms as T
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dtype = torch.float16 if cfg.fp16 else torch.float32

        # old adapter
        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []
        self.mix_b = cfg.mix_bias

    def encode_text(self, text, prompt=False):
        x = self.token_embedding(text).type(self.clip_type)
        x = x + self.positional_embedding.type(self.clip_type)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image):
        # Always use CLIP visual
        clip_feat = self.visual(image.to(self.clip_type)).float()
        if self.use_dino:
            dino_feat = self.dino(image.float())
            return torch.cat([clip_feat, dino_feat], dim=-1)  # [B, 1280]
        return clip_feat

    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)

        with torch.no_grad():
            image_features = self.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.type(self.dtype).detach()).type(self.clip_type)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)
        
        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def adaptation(self, task_id, threshold=0):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self.get_class_name_features()
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.queue_empty = True
        self.hard_pairs = None
        if task_id > 0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-len(self.class_ids_per_task[task_id])]):
                diff = torch.cdist(self.class_name_features[-len(self.class_ids_per_task[task_id]):].type(torch.float32), class_name_feature.unsqueeze(0).type(torch.float32)).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_new_class = torch.unique(indices[:,1]) + self.cfg.initial_increment+(task_id-1) * self.cfg.increment
            num_hard_class = self.hard_new_class.shape[0]
            self.hard_pairs = indices
            self.hard_pairs[:,1] = self.hard_pairs[:,1]+self.cfg.initial_increment+(task_id-1) * self.cfg.increment

    def get_old_edge_samples(self, batch_size):
        random_select = torch.randperm(self.old_edge_samples.shape[0])[:batch_size]
        return self.old_edge_samples[random_select], self.old_edge_samples_labels[random_select], self.old_edge_samples_nearest_labels[random_select]

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l).squeeze()
            class_data = features[index]
            if class_data.dim() == 1:
                class_data = class_data.unsqueeze(0)
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            if distance.dim() == 0:
                distance = distance.unsqueeze(0)
            max_distance = torch.sort(distance)[0][-min(10, len(distance)):]
            self.class_edge_distance.append((max_distance.mean()-max_distance.min(), max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)

    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            U_old, S_old, V_old = torch.linalg.svd(weight_old, full_matrices=False)
            # U_old: [out, k], S_old: [k], V_old: [k, in] where k=min(out, in)
            P_new = U_old.T @ weight_new  # [k, in]
            S_diag_V = torch.diag(S_old) @ V_old  # [k, in]
            dist = (P_new - S_diag_V).abs()
            mask = dist / (dist.max() + 1e-8)
            mask += self.mix_b
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + S_diag_V * (1 - mask)
            weight = U_old @ right
            self.adapter.weight.data = weight
            return


class DomainIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False) -> None:
        super().__init__()
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.text_tokens = None
        self.prompt_template = cfg.prompt_template
        self.device = device

    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    def tokenize(self, class_names):
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in class_names]
        ).to(self.device)



class TaskAgnosticCLIP(nn.Module):
    pass



def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        use_dino = getattr(cfg, 'use_dino', False)
        if use_dino:
            return ClassIncrementalDINO(cfg, device)
        return ClassIncrementalCLIP(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncrementalCLIP(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnosticCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    
