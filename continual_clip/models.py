import os


import copy
import pdb
from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn

from .utils import get_class_ids_per_task, get_class_names, get_aircraft_descriptive_name
from .lora import LoRALinear, inject_lora, get_lora_state_dict, load_lora_state_dict

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
        # Use descriptive names for FGVC-Aircraft if available
        if self.cfg.dataset == "fgvc_aircraft":
            descriptive_names = [get_aircraft_descriptive_name(c) for c in self.current_class_names]
            self.text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in descriptive_names]
            ).to(self.device)
        else:
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




class DINOv2Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DINOv2MLP(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DINOv2Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOv2Attention(dim, num_heads)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DINOv2MLP(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DINOv2Encoder(nn.Module):
    """Native DINOv2 ViT-B/14 encoder that matches the official checkpoint structure."""
    def __init__(self, weights_path=None, device="cuda", input_size=224):
        super().__init__()
        self.embed_dim = 768
        self.patch_size = 14
        self.input_size = input_size
        num_patches = (input_size // self.patch_size) ** 2  # 256 for 224x224

        self.patch_embed = nn.Sequential()
        self.patch_embed.proj = nn.Conv2d(3, 768, kernel_size=14, stride=14)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, 768))

        self.blocks = nn.ModuleList([DINOv2Block(768, 12, 4.0) for _ in range(12)])
        self.norm = nn.LayerNorm(768)

        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            # Remove mask_token (not needed for inference)
            state_dict.pop("mask_token", None)

            # Interpolate pos_embed from training resolution to input resolution
            pos_embed = state_dict["pos_embed"]  # [1, 1370, 768]
            cls_pe = pos_embed[:, :1]  # [1, 1, 768]
            patch_pe = pos_embed[:, 1:]  # [1, 1369, 768]
            old_size = int(patch_pe.shape[1] ** 0.5)  # 37
            new_size = input_size // self.patch_size  # 16
            patch_pe = patch_pe.reshape(1, old_size, old_size, 768).permute(0, 3, 1, 2)
            patch_pe = nn.functional.interpolate(
                patch_pe.float(), size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_size * new_size, 768)
            state_dict["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)

            # Rename ls1.gamma/ls2.gamma -> blocks.X.ls1/ls2
            new_sd = {}
            for k, v in state_dict.items():
                if ".ls1.gamma" in k:
                    new_sd[k.replace(".ls1.gamma", ".ls1")] = v
                elif ".ls2.gamma" in k:
                    new_sd[k.replace(".ls2.gamma", ".ls2")] = v
                else:
                    new_sd[k] = v

            # Rename patch_embed.proj -> patch_embed.proj (already matches via Sequential)
            missing, unexpected = self.load_state_dict(new_sd, strict=False)
            print(f"DINOv2 loaded: {len(new_sd)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
            if missing:
                print(f"  Missing: {missing}")
            if unexpected:
                print(f"  Unexpected: {unexpected}")

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed.proj(x)  # [B, 768, H/14, W/14]
        x = x.flatten(2).transpose(1, 2)  # [B, N, 768]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


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
            self.dino = DINOv2Encoder(dino_weights, device="cpu")
            # Inject LoRA BEFORE moving to device so params are registered
            lora_rank = getattr(cfg, 'lora_rank', 16)
            lora_targets = getattr(cfg, 'lora_targets', ['qkv', 'proj'])
            if isinstance(lora_targets, str):
                lora_targets = [lora_targets]
            self.lora_params = inject_lora(self.dino, rank=lora_rank, alpha=lora_rank*2, targets=tuple(lora_targets))
            self.dino = self.dino.to(device)  # now move everything (including LoRA) to device
            # Update lora_params references to point to CUDA tensors
            self.lora_params = [p for m in self.dino.modules() if isinstance(m, LoRALinear) for p in [m.lora_A, m.lora_B]]
            self.old_lora_state = None
            visual_dim = 512 + 768
            print(f"Using CLIP+DINOv2 fusion (dim={visual_dim}) with LoRA rank={lora_rank}")
        else:
            visual_dim = 512
            print(f"DINOv2 not found, using CLIP only (dim={visual_dim})")
        
        # Projection: concat_dim -> 512 (CLIP text space)
        self.adapter = nn.Linear(visual_dim, 512, bias=False, device=device)
        
        # Keep CLIP transforms as main transforms (used by DataLoader)
        # Store DINOv2-specific normalization for dual-path encoding
        from torchvision import transforms as T
        self.dino_normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.clip_normalize = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # Use a shared preprocessing (resize + to_tensor) without normalization
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
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
    
    def encode_image(self, image, allow_lora_grad=False):
        # Dual-path normalization: each encoder gets its own normalization
        image_float = image.float()
        with torch.no_grad():
            clip_input = self.clip_normalize(image_float).to(self.clip_type)
            clip_feat = self.visual(clip_input).float()
        if self.use_dino:
            dino_input = self.dino_normalize(image_float)
            if allow_lora_grad:
                # Allow gradients through LoRA parameters
                dino_feat = self.dino(dino_input)
            else:
                with torch.no_grad():
                    dino_feat = self.dino(dino_input)
            return torch.cat([clip_feat, dino_feat.float()], dim=-1)  # [B, 1280]
        return clip_feat

    @torch.no_grad()
    def get_class_name_features(self):
        class_name_features = self.encode_text(self.text_tokens)
        return class_name_features.type(torch.float32)

    def forward(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        with torch.no_grad():
            text_features = self.encode_text(self.text_tokens)

        # Allow LoRA gradients during training for real images
        image_features = self.encode_image(image, allow_lora_grad=self.training)
        original_image_features = image_features.clone().detach()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)

        image_features = self.adapter(image_features.type(self.dtype)).type(self.clip_type)

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
        # Use descriptive names for FGVC-Aircraft if available
        if self.cfg.dataset == "fgvc_aircraft":
            descriptive_names = [get_aircraft_descriptive_name(c) for c in self.current_class_names]
            self.text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in descriptive_names]
            ).to(self.device)
        else:
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
            # Save old LoRA state for replay comparison
            if self.use_dino and hasattr(self, 'lora_params'):
                self.old_lora_state = get_lora_state_dict(self.dino)
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
    
