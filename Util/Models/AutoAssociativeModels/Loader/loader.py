import os
import re
import random
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader

FILENAME_RE = re.compile(r"^(?P<listener>[^_]+)_(?P<session>[^_]+)_(?P<audio>[^_.]+)\.pkl$")


@dataclass
class Instance:
    listener_id: str
    session_id: str
    audio_id: str
    position: int
    label: str
    eeg: any  # output of preprocess_fn, whatever shape you make it


def load_instances(pkl_dir: str, listener_ids: Sequence[str], preprocess_fn) -> List[Instance]:
    # only loads files belonging to the given listener_ids
    listener_ids = set(listener_ids)
    instances = []
    for fname in sorted(os.listdir(pkl_dir)):
        if not fname.endswith(".pkl"):
            continue
        m = FILENAME_RE.match(fname)
        if m is None or m.group("listener") not in listener_ids:
            continue

        with open(os.path.join(pkl_dir, fname), "rb") as f:
            d = pickle.load(f)

        listener_id = d["listener_id"]
        session_id = d["session_id"]
        audio_id = d["audio_id"]
        eeg_list = d["eeg_data"]
        labels = d["label"]

        for pos, (eeg, label) in enumerate(zip(eeg_list, labels)):
            eeg = preprocess_fn(eeg)
            instances.append(Instance(listener_id, session_id, audio_id, pos, label, eeg))
    return instances


LabelListenerIndex = Dict[str, Dict[str, List[int]]]


def build_index(instances: Sequence[Instance]) -> LabelListenerIndex:
    # label -> listener_id -> [instance idx]
    index = defaultdict(lambda: defaultdict(list))
    for i, inst in enumerate(instances):
        index[inst.label][inst.listener_id].append(i)
    return {label: dict(per_listener) for label, per_listener in index.items()}


def make_sample(x_inst: Instance, y_inst: Instance) -> dict:
    return {
        "x": torch.as_tensor(x_inst.eeg, dtype=torch.float32),
        "y": torch.as_tensor(y_inst.eeg, dtype=torch.float32),
        "x_listener_id": x_inst.listener_id,
        "y_listener_id": y_inst.listener_id,
        "x_session_id": x_inst.session_id,
        "y_session_id": y_inst.session_id,
        "x_audio_id": x_inst.audio_id,
        "y_audio_id": y_inst.audio_id,
        "label": x_inst.label,
    }


class CrossSubjectEEGDataset(Dataset):
    # for every instance (as X), pairs it with every other-subject same-label
    # instance (as Y), across all sentences. max_pairs_per_x=None -> use all
    # combinations, else randomly cap to that many per X.
    def __init__(self, instances: Sequence[Instance], index: LabelListenerIndex,
                 max_pairs_per_x: Optional[int] = None, seed: int = 0):
        self.instances = instances
        rng = random.Random(seed)
        self.pairs = []

        for x_idx, x_inst in enumerate(instances):
            per_listener = index[x_inst.label]
            candidates = [
                y_idx
                for lid, idxs in per_listener.items()
                if lid != x_inst.listener_id
                for y_idx in idxs
            ]
            if max_pairs_per_x is not None and len(candidates) > max_pairs_per_x:
                candidates = rng.sample(candidates, max_pairs_per_x)
            for y_idx in candidates:
                self.pairs.append((x_idx, y_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_idx, y_idx = self.pairs[idx]
        return make_sample(self.instances[x_idx], self.instances[y_idx])


def build_datasets(pkl_dir, train_listeners, valid_listeners, preprocess_fn,
                    max_pairs_per_x=None, seed=0):
    train_instances = load_instances(pkl_dir, train_listeners, preprocess_fn)
    valid_instances = load_instances(pkl_dir, valid_listeners, preprocess_fn)

    train_index = build_index(train_instances)
    valid_index = build_index(valid_instances)

    train_ds = CrossSubjectEEGDataset(train_instances, train_index, max_pairs_per_x, seed)
    valid_ds = CrossSubjectEEGDataset(valid_instances, valid_index, max_pairs_per_x, seed)
    return train_ds, valid_ds


def build_dataloaders(train_ds, valid_ds, batch_size=64, num_workers=0):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


if __name__ == "__main__":
    PKL_DIR = "Data/52_pkl/"
    TRAIN_LISTENERS = ["01C100104", "021100103", "017100101"]
    VALID_LISTENERS = ["016100096"]

    def preprocess_fn(eeg, target_len=10):
        # pad/truncate variable-length [T,128] -> [target_len,128]
        import numpy as np
        T = eeg.shape[0]
        if T == target_len:
            return eeg
        if T > target_len:
            return eeg[:target_len]
        return np.pad(eeg, ((0, target_len - T), (0, 0)), mode="constant")

    train_ds, valid_ds = build_datasets(
        PKL_DIR, TRAIN_LISTENERS, VALID_LISTENERS, preprocess_fn, max_pairs_per_x=None
    )
    train_loader, valid_loader = build_dataloaders(train_ds, valid_ds, batch_size=64)

    batch = next(iter(train_loader))
    print(len(train_loader))
    print(batch)