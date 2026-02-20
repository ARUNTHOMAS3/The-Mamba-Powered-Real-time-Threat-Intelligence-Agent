"""
Dataset Factory
Centralizes dataset selection logic for the benchmark runner.
"""

from datasets.cicids2017_loader import CICIDS2017Dataset
from datasets.unswnb15_loader import UNSWNB15Dataset
from datasets.cicids2018_loader import CICIDS2018Dataset


DATASET_REGISTRY = {
    'CICIDS2017': {
        'class': CICIDS2017Dataset,
        'root_dir': 'data/raw/CICIDS2017',
        'description': 'Canadian Institute for Cybersecurity IDS 2017 (Monday + Friday subset)',
    },
    'UNSW-NB15': {
        'class': UNSWNB15Dataset,
        'root_dir': 'data/raw/UNSW-NB15',
        'description': 'UNSW-NB15 Network Intrusion Dataset',
    },
    'CIC-IDS2018': {
        'class': CICIDS2018Dataset,
        'root_dir': 'data/raw/CIC-IDS2018',
        'description': 'Canadian Institute for Cybersecurity IDS 2018',
    },
}


def get_dataset(name, split, seq_len=50, binary=True):
    """
    Factory function to get a dataset by name.
    
    Args:
        name (str): Dataset name (CICIDS2017, UNSW-NB15, CIC-IDS2018)
        split (str): Data split (train, val, test)
        seq_len (int): Sequence length for temporal windows
        binary (bool): Whether to use binary classification
    
    Returns:
        Dataset instance
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    
    info = DATASET_REGISTRY[name]
    return info['class'](
        root_dir=info['root_dir'],
        split=split,
        binary=binary,
        seq_len=seq_len
    )


def list_datasets():
    """List all available datasets with descriptions."""
    for name, info in DATASET_REGISTRY.items():
        print(f"  {name}: {info['description']}")
    return list(DATASET_REGISTRY.keys())
