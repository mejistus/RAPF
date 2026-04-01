"""
Create 3 merged dataset variants by copying and reorganizing classes.
Mild: merge only identical aircraft (DC-3/C-47, etc.) — 3 merges → 97 classes
Moderate: merge same-series sub-variants — 12 merges → 72 classes  
Aggressive: merge all confusable families — 22 merges → 52 classes
"""
import os
import shutil
from collections import defaultdict

SRC = "../data/fgvc_aircraft"

# ==================== Merge Schemes ====================

MILD = {
    # Only merge truly identical aircraft (military/civil variants)
    "DC-3_C-47": ["DC-3", "C-47"],                    # same aircraft
    "BAE_146": ["BAE_146-200", "BAE_146-300"],         # only fuselage length
    "ERJ_135_145": ["ERJ_135", "ERJ_145"],             # same platform
}
# 100 - 3 merges (each saves 1 class) = 97 classes

MODERATE = {
    **MILD,
    # Merge sub-variants within same series
    "737_Classic": ["737-200", "737-300", "737-400", "737-500"],
    "737_NG": ["737-600", "737-700", "737-800", "737-900"],
    "747": ["747-100", "747-200", "747-300", "747-400"],
    "767": ["767-200", "767-300", "767-400"],
    "777": ["777-200", "777-300"],
    "A320_Family": ["A318", "A319", "A320", "A321"],
    "A330": ["A330-200", "A330-300"],
    "A340": ["A340-200", "A340-300", "A340-500", "A340-600"],
    "MD_Series": ["MD-80", "MD-87", "MD-90"],
}
# Removes: 3+6+6+5+4+2+5+3+5+4 = many → count precisely below

AGGRESSIVE = {
    **MODERATE,
    "Boeing_717_DC9": ["Boeing_717", "DC-9-30"],           # 717 = DC-9 derivative
    "MD_DC_Wide": ["MD-11", "DC-10"],                      # tri-engine wide-body
    "A300_A310": ["A300B4", "A310"],                       # early Airbus wide-body
    "757": ["757-200", "757-300"],
    "707_DC8": ["707-320", "DC-8"],                        # 4-engine early jets
    "ATR": ["ATR-42", "ATR-72"],                           # twin turboprop
    "CRJ": ["CRJ-200", "CRJ-700", "CRJ-900"],            # regional jets
    "Embraer_EJet": ["E-170", "E-190", "E-195"],          # E-Jet family
    "Fokker": ["Fokker_50", "Fokker_70", "Fokker_100"],   # Fokker family
    "Gulfstream": ["Gulfstream_IV", "Gulfstream_V"],      # biz jets
    "Saab": ["Saab_2000", "Saab_340"],                    # Swedish turboprops
    "DHC8": ["DHC-8-100", "DHC-8-300"],                   # Dash 8
    "Falcon": ["Falcon_2000", "Falcon_900"],              # Dassault biz jets
}


def count_classes(scheme, total=100):
    merged_classes = set()
    for members in scheme.values():
        merged_classes.update(members)
    new_count = total - len(merged_classes) + len(scheme)
    return new_count


def create_dataset(scheme, name):
    dst = os.path.join(os.path.dirname(SRC), name)
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Build reverse map: old_class -> new_class
    remap = {}
    for new_name, old_names in scheme.items():
        for old in old_names:
            remap[old] = new_name

    for split in ["train", "test"]:
        src_split = os.path.join(SRC, split)
        dst_split = os.path.join(dst, split)
        os.makedirs(dst_split, exist_ok=True)

        for cls_dir in sorted(os.listdir(src_split)):
            src_cls = os.path.join(src_split, cls_dir)
            if not os.path.isdir(src_cls):
                continue
            # Determine target class name
            new_cls = remap.get(cls_dir, cls_dir)
            dst_cls = os.path.join(dst_split, new_cls)
            os.makedirs(dst_cls, exist_ok=True)

            # Copy files (prefix with original class to avoid name collisions)
            for fname in os.listdir(src_cls):
                src_file = os.path.join(src_cls, fname)
                if cls_dir != new_cls:
                    # Prefix to avoid collision
                    dst_file = os.path.join(dst_cls, f"{cls_dir}__{fname}")
                else:
                    dst_file = os.path.join(dst_cls, fname)
                shutil.copy2(src_file, dst_file)

    # Count
    train_classes = sorted(os.listdir(os.path.join(dst, "train")))
    test_classes = sorted(os.listdir(os.path.join(dst, "test")))
    train_total = sum(len(os.listdir(os.path.join(dst, "train", c))) for c in train_classes)
    test_total = sum(len(os.listdir(os.path.join(dst, "test", c))) for c in test_classes)
    return len(train_classes), train_total, test_total


if __name__ == "__main__":
    print(f"Source: {SRC}")
    print(f"  Train classes: {len(os.listdir(os.path.join(SRC, 'train')))}")
    print()

    for scheme, name, label in [
        (MILD, "fgvc_mild", "Mild"),
        (MODERATE, "fgvc_moderate", "Moderate"),
        (AGGRESSIVE, "fgvc_aggressive", "Aggressive"),
    ]:
        expected = count_classes(scheme)
        nc, nt, ntest = create_dataset(scheme, name)
        merged_list = [(k, v) for k, v in scheme.items() if k not in MILD or label != "Moderate"]
        print(f"{label}: {nc} classes (from 100), {nt} train, {ntest} test")
        print(f"  Merges ({len(scheme)}):")
        for new_name, old_names in scheme.items():
            print(f"    {new_name} <- {old_names}")
        print()
