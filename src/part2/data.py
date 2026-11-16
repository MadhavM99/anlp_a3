from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected

SPLITS = ("train", "val", "test")


class Standardize(BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, data):
        data.x = (data.x - self.mean) / self.std
        return data


def load_splits(root, name, feature, standardize=False):
    splits = {split: UPFD(root, name, feature, split) for split in SPLITS}
    if standardize:
        x = splits["train"]._data.x
        transform = Standardize(x.mean(dim=0), x.std(dim=0).clamp_min(1e-6))
        for dataset in splits.values():
            dataset.transform = transform
    return splits


def make_loaders(splits, batch_size, generator):
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", generator=generator)
        for split, dataset in splits.items()
    }


def describe(splits):
    sample = splits["train"][0]
    x = splits["train"]._data.x
    return {
        "x_absmax": float(x.abs().max()),
        "x_mean_of_std": float(x.std(dim=0).mean()),
        "sizes": {split: len(dataset) for split, dataset in splits.items()},
        "num_features": splits["train"].num_features,
        "stored_undirected": bool(is_undirected(sample.edge_index, num_nodes=sample.num_nodes)),
        "root_is_source": bool((sample.edge_index[0] == 0).any()),
    }
