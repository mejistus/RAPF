"""PK Batch Sampler: sample P classes with K images each per batch."""
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    Sample P classes, K images per class per batch.
    Batch size = P * K.
    Each epoch traverses all samples at least once.
    """
    def __init__(self, labels, p=24, k=2):
        self.labels = labels
        self.p = p
        self.k = k
        self.batch_size = p * k

        # Build class -> indices mapping
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)

        # Ensure we have enough classes
        assert self.p <= self.num_classes, f"P={p} > num_classes={self.num_classes}"

        # Estimate length: enough batches to see all samples ~once
        self.num_batches = max(len(labels) // self.batch_size, 1)

    def __iter__(self):
        # Shuffle indices within each class
        class_indices = {}
        for c in self.classes:
            inds = self.class_to_indices[c].copy()
            random.shuffle(inds)
            class_indices[c] = inds
        class_pointers = {c: 0 for c in self.classes}

        # Shuffle class order each epoch
        class_pool = self.classes.copy()
        random.shuffle(class_pool)
        pool_ptr = 0

        for _ in range(self.num_batches):
            batch = []
            # Pick P classes
            selected_classes = []
            for _ in range(self.p):
                if pool_ptr >= len(class_pool):
                    random.shuffle(class_pool)
                    pool_ptr = 0
                selected_classes.append(class_pool[pool_ptr])
                pool_ptr += 1

            # Pick K samples per class
            for c in selected_classes:
                inds = class_indices[c]
                ptr = class_pointers[c]
                picked = []
                for _ in range(self.k):
                    if ptr >= len(inds):
                        random.shuffle(inds)
                        ptr = 0
                    picked.append(inds[ptr])
                    ptr += 1
                class_pointers[c] = ptr
                batch.extend(picked)

            yield batch

    def __len__(self):
        return self.num_batches
