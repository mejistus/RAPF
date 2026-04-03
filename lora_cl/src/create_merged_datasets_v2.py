"""
V2: Corrected merge schemes — ONLY merge sub-variants of the SAME aircraft type.
Never merge different aircraft that merely look similar.

Principle: A→B merge is valid ONLY if A and B share the same type certificate
or are universally recognized as the same aircraft type with different fuselage lengths.
"""
import os
import shutil

SRC = "../data/fgvc_aircraft"

# ==================== CORRECTED Merge Schemes ====================

MILD = {
    # Only truly identical aircraft (military/civil version of same airframe)
    "DC-3_C-47": ["DC-3", "C-47"],           # C-47 IS DC-3 (military designation)
    "BAE_146": ["BAE_146-200", "BAE_146-300"],  # same type, different fuselage length
}
# 100 - 2 = 98 classes

MODERATE = {
    **MILD,
    # Same type certificate, different fuselage lengths only
    "737_Classic": ["737-200", "737-300", "737-400", "737-500"],   # 737 Classic family
    "737_NG": ["737-600", "737-700", "737-800", "737-900"],        # 737 Next Gen family
    "747": ["747-100", "747-200", "747-300", "747-400"],           # 747 variants
    "757": ["757-200", "757-300"],                                  # 757 variants
    "767": ["767-200", "767-300", "767-400"],                      # 767 variants
    "777": ["777-200", "777-300"],                                  # 777 variants
    "A320_Family": ["A318", "A319", "A320", "A321"],               # A320 family
    "A330": ["A330-200", "A330-300"],                               # A330 variants
    "A340": ["A340-200", "A340-300", "A340-500", "A340-600"],     # A340 variants
    "MD_80_Series": ["MD-80", "MD-87", "MD-90"],                   # MD-80 series (same base)
    "ERJ": ["ERJ_135", "ERJ_145"],                                 # ERJ family (same platform)
    "DHC8": ["DHC-8-100", "DHC-8-300"],                            # Dash 8 variants
    "E_Jet": ["E-170", "E-190", "E-195"],                         # Embraer E-Jet family
    "CRJ": ["CRJ-200", "CRJ-700", "CRJ-900"],                    # CRJ family
}
# These are all genuinely same-type sub-variants

AGGRESSIVE = {
    **MODERATE,
    # Still same-type but slightly more debatable
    "ATR": ["ATR-42", "ATR-72"],               # same manufacturer, same platform, different length
    "Fokker_Jet": ["Fokker_70", "Fokker_100"], # same type (F28 derivative), different length
    # NOTE: Fokker_50 is a TURBOPROP, stays separate!
    "Saab_340_2000": ["Saab_340", "Saab_2000"],  # both Saab turboprops, same lineage
}
# Fokker_50, Gulfstream_IV/V, Falcon_2000/900, 707, DC-8, DC-10, MD-11,
# A300B4, A310, Boeing_717, DC-9-30 all stay SEPARATE (different aircraft)


def count_classes(scheme, total=100):
    merged = set()
    for members in scheme.values():
        merged.update(members)
    return total - len(merged) + len(scheme)


def create_dataset(scheme, name):
    dst = os.path.join(os.path.dirname(SRC), name)
    if os.path.exists(dst):
        shutil.rmtree(dst)

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
            if not os.path.isdir(src_cls): continue
            new_cls = remap.get(cls_dir, cls_dir)
            dst_cls = os.path.join(dst_split, new_cls)
            os.makedirs(dst_cls, exist_ok=True)
            for fname in os.listdir(src_cls):
                src_file = os.path.join(src_cls, fname)
                if cls_dir != new_cls:
                    dst_file = os.path.join(dst_cls, f"{cls_dir}__{fname}")
                else:
                    dst_file = os.path.join(dst_cls, fname)
                shutil.copy2(src_file, dst_file)

    train_cls = sorted(os.listdir(os.path.join(dst, "train")))
    train_n = sum(len(os.listdir(os.path.join(dst, "train", c))) for c in train_cls)
    test_n = sum(len(os.listdir(os.path.join(dst, "test", c))) for c in train_cls)
    return len(train_cls), train_n, test_n


if __name__ == "__main__":
    print("=== V2 Corrected Merge (only same-type sub-variants) ===\n")

    for scheme, name, label in [
        (MILD, "fgvc_mild_v2", "Mild"),
        (MODERATE, "fgvc_moderate_v2", "Moderate"),
        (AGGRESSIVE, "fgvc_aggressive_v2", "Aggressive"),
    ]:
        nc, nt, ntest = create_dataset(scheme, name)
        print(f"{label}: {nc} classes, {nt} train, {ntest} test")
        print(f"  Merges ({len(scheme)}):")
        for new_name, old_names in scheme.items():
            print(f"    {new_name} <- {old_names}")

        # List classes that STAY SEPARATE
        merged_originals = set()
        for members in scheme.values():
            merged_originals.update(members)
        kept = [c for c in sorted(os.listdir(os.path.join(SRC, "train"))) if c not in merged_originals]
        print(f"  Kept separate ({len(kept)}): {kept[:5]}... ({len(kept)} total)")
        print()
