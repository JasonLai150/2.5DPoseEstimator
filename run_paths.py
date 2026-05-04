from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
FIT3D_2D_DIR = ARTIFACTS_DIR / "fit3d_2d"
COCO_2D_DIR = ARTIFACTS_DIR / "coco_2d"
COCO_ACAE_DIR = ARTIFACTS_DIR / "coco_acae"
VISUALIZATION_DIR = ARTIFACTS_DIR / "visualizations"
VITPOSE_CHECKPOINT_DIR = CHECKPOINT_DIR / "vitpose"
VITPOSE_B_CHECKPOINT_PATH = VITPOSE_CHECKPOINT_DIR / "vitpose_base_coco_256x192.pth"

LEGACY_CHECKPOINT_DIR = REPO_ROOT / "checkpoint"
LEGACY_ACAE_CHECKPOINT_DIR = REPO_ROOT / "acae_data" / "checkpoints"

VDP3D_CHECKPOINT_DIR = CHECKPOINT_DIR
ACAE_CHECKPOINT_PATH = CHECKPOINT_DIR / "h36_fit_checkpoint.pth"
MIXED_CHECKPOINT_PATH = CHECKPOINT_DIR / "mixed_finetuned_model.bin"
H36M_BRIDGE_CHECKPOINT_PATH = CHECKPOINT_DIR / "vdp3d_h36_fit_bridge.bin"
PRETRAINED_H36M_CHECKPOINT_PATH = CHECKPOINT_DIR / "epoch_120.bin"
VISUALIZATION_CHECKPOINT_PATH = CHECKPOINT_DIR / "epoch_110.bin"


def ensure_artifact_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIT3D_2D_DIR.mkdir(parents=True, exist_ok=True)
    COCO_2D_DIR.mkdir(parents=True, exist_ok=True)
    COCO_ACAE_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    VITPOSE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def first_existing_path(*paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    return Path(paths[0])
