__all__ = [
    "set_seed",
    "dict2cfg",
    "get_dataloader",
    "get_scheduler",
    "HFAudioClassifier",
]


def __getattr__(name):
    if name == "set_seed":
        from sonics.utils.seed import set_seed

        return set_seed
    if name == "dict2cfg":
        from sonics.utils.config import dict2cfg

        return dict2cfg
    if name == "get_dataloader":
        from sonics.utils.dataset import get_dataloader

        return get_dataloader
    if name == "get_scheduler":
        from sonics.utils.scheduler import get_scheduler

        return get_scheduler
    if name == "HFAudioClassifier":
        from sonics.models.hf_model import HFAudioClassifier

        return HFAudioClassifier
    raise AttributeError(f"module 'sonics' has no attribute {name!r}")
