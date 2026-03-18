from .base import PoseEstimatorBase
from .dstformer import DSTformer
from .videopose import VideoPose3D
from .lora import LoRALinear, apply_lora_to_model, freeze_non_lora, count_lora_parameters
from .pretrained import load_pretrained_model, MotionBERTWrapper, APTPoseWrapper

__all__ = [
    "PoseEstimatorBase",
    "DSTformer",
    "VideoPose3D",
    "LoRALinear",
    "apply_lora_to_model",
    "freeze_non_lora",
    "count_lora_parameters",
    "load_pretrained_model",
    "MotionBERTWrapper",
    "APTPoseWrapper",
    "create_model",
]


def create_model(cfg) -> PoseEstimatorBase:
    """
    Factory function to create a model from config.

    Args:
        cfg: Configuration with model settings

    Returns:
        Initialized model
    """
    model_name = cfg.model.get('name', 'dstformer').lower()

    if model_name == 'dstformer':
        pretrained_path = cfg.model.get('pretrained_path', None)
        return DSTformer(cfg, pretrained_path=pretrained_path)
    elif model_name == 'videopose' or model_name == 'videopose3d':
        pretrained_path = cfg.model.get('pretrained_path', None)
        return VideoPose3D(cfg, pretrained_path=pretrained_path)
    elif model_name == 'motionbert':
        checkpoint = cfg.model.get('pretrained_path', None)
        return MotionBERTWrapper(cfg, checkpoint_path=checkpoint)
    elif model_name == 'aptpose':
        checkpoint = cfg.model.get('pretrained_path', None)
        return APTPoseWrapper(cfg, checkpoint_path=checkpoint)
    else:
        raise ValueError(f"Unknown model: {model_name}")
