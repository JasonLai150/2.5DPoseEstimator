"""
Skeleton Mapper: COCO 17-keypoint to Human3.6M skeleton conversion.

Handles index reordering, dropping incompatible joints (facial keypoints),
and interpolating missing nodes (pelvis, spine, thorax).
"""

import numpy as np

# COCO 17-keypoint indices
COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# Human3.6M 17-joint indices
H36M_JOINTS = {
    'pelvis': 0,        # Interpolated from hips
    'right_hip': 1,
    'right_knee': 2,
    'right_ankle': 3,
    'left_hip': 4,
    'left_knee': 5,
    'left_ankle': 6,
    'spine': 7,         # Interpolated
    'thorax': 8,        # Interpolated
    'neck_nose': 9,
    'head': 10,
    'left_shoulder': 11,
    'left_elbow': 12,
    'left_wrist': 13,
    'right_shoulder': 14,
    'right_elbow': 15,
    'right_wrist': 16,
}

# Direct mappings: H36M index -> COCO index
COCO_TO_H36M_DIRECT = {
    H36M_JOINTS['right_hip']: COCO_KEYPOINTS['right_hip'],
    H36M_JOINTS['right_knee']: COCO_KEYPOINTS['right_knee'],
    H36M_JOINTS['right_ankle']: COCO_KEYPOINTS['right_ankle'],
    H36M_JOINTS['left_hip']: COCO_KEYPOINTS['left_hip'],
    H36M_JOINTS['left_knee']: COCO_KEYPOINTS['left_knee'],
    H36M_JOINTS['left_ankle']: COCO_KEYPOINTS['left_ankle'],
    H36M_JOINTS['neck_nose']: COCO_KEYPOINTS['nose'],
    H36M_JOINTS['left_shoulder']: COCO_KEYPOINTS['left_shoulder'],
    H36M_JOINTS['left_elbow']: COCO_KEYPOINTS['left_elbow'],
    H36M_JOINTS['left_wrist']: COCO_KEYPOINTS['left_wrist'],
    H36M_JOINTS['right_shoulder']: COCO_KEYPOINTS['right_shoulder'],
    H36M_JOINTS['right_elbow']: COCO_KEYPOINTS['right_elbow'],
    H36M_JOINTS['right_wrist']: COCO_KEYPOINTS['right_wrist'],
}

# Interpolation rules: H36M index -> (COCO indices, weights)
COCO_TO_H36M_INTERPOLATE = {
    # Pelvis: midpoint of left and right hip
    H36M_JOINTS['pelvis']: (
        [COCO_KEYPOINTS['left_hip'], COCO_KEYPOINTS['right_hip']],
        [0.5, 0.5]
    ),
    # Spine: 2/3 from pelvis toward thorax (approximated from shoulders/hips)
    H36M_JOINTS['spine']: (
        [COCO_KEYPOINTS['left_hip'], COCO_KEYPOINTS['right_hip'],
         COCO_KEYPOINTS['left_shoulder'], COCO_KEYPOINTS['right_shoulder']],
        [0.25, 0.25, 0.25, 0.25]  # Center of torso
    ),
    # Thorax: midpoint of shoulders
    H36M_JOINTS['thorax']: (
        [COCO_KEYPOINTS['left_shoulder'], COCO_KEYPOINTS['right_shoulder']],
        [0.5, 0.5]
    ),
    # Head: approximate from nose and ears (or just nose if ears unavailable)
    H36M_JOINTS['head']: (
        [COCO_KEYPOINTS['nose'], COCO_KEYPOINTS['left_ear'], COCO_KEYPOINTS['right_ear']],
        [0.34, 0.33, 0.33]
    ),
}


def coco_to_h36m(coco_keypoints: np.ndarray, confidence: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert COCO 17-keypoint format to Human3.6M 17-joint format.

    Args:
        coco_keypoints: Array of shape (..., 17, D) where D is 2 or 3
        confidence: Optional confidence scores of shape (..., 17)

    Returns:
        h36m_joints: Array of shape (..., 17, D) in H36M format
        h36m_confidence: Confidence scores of shape (..., 17)
    """
    input_shape = coco_keypoints.shape
    D = input_shape[-1]  # 2D or 3D coordinates

    # Reshape to (N, 17, D) for processing
    coco_flat = coco_keypoints.reshape(-1, 17, D)
    N = coco_flat.shape[0]

    if confidence is None:
        conf_flat = np.ones((N, 17))
    else:
        conf_flat = confidence.reshape(-1, 17)

    # Initialize output
    h36m_joints = np.zeros((N, 17, D), dtype=coco_flat.dtype)
    h36m_conf = np.zeros((N, 17), dtype=conf_flat.dtype)

    # Apply direct mappings
    for h36m_idx, coco_idx in COCO_TO_H36M_DIRECT.items():
        h36m_joints[:, h36m_idx] = coco_flat[:, coco_idx]
        h36m_conf[:, h36m_idx] = conf_flat[:, coco_idx]

    # Apply interpolations
    for h36m_idx, (coco_indices, weights) in COCO_TO_H36M_INTERPOLATE.items():
        weights = np.array(weights)
        for i, (coco_idx, w) in enumerate(zip(coco_indices, weights)):
            h36m_joints[:, h36m_idx] += w * coco_flat[:, coco_idx]
        # Confidence is minimum of constituent joints
        h36m_conf[:, h36m_idx] = np.min(conf_flat[:, coco_indices], axis=-1)

    # Restore original batch shape
    output_shape = input_shape[:-2] + (17, D)
    conf_shape = input_shape[:-2] + (17,)

    return h36m_joints.reshape(output_shape), h36m_conf.reshape(conf_shape)


# Bone connections for each skeleton (for visualization/loss computation)
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),      # Right leg: pelvis -> hip -> knee -> ankle
    (0, 4), (4, 5), (5, 6),      # Left leg: pelvis -> hip -> knee -> ankle
    (0, 7), (7, 8), (8, 9),      # Spine: pelvis -> spine -> thorax -> neck
    (9, 10),                      # Head: neck -> head
    (8, 11), (11, 12), (12, 13), # Left arm: thorax -> shoulder -> elbow -> wrist
    (8, 14), (14, 15), (15, 16), # Right arm: thorax -> shoulder -> elbow -> wrist
]

# Bilateral pairs for symmetry loss
H36M_BILATERAL_PAIRS = [
    # (left_bone, right_bone) where bone is (joint1, joint2)
    ((4, 5), (1, 2)),   # Hip to knee
    ((5, 6), (2, 3)),   # Knee to ankle
    ((11, 12), (14, 15)), # Shoulder to elbow
    ((12, 13), (15, 16)), # Elbow to wrist
]


def compute_bone_lengths(joints: np.ndarray, bones: list[tuple[int, int]]) -> np.ndarray:
    """
    Compute bone lengths for a skeleton.

    Args:
        joints: Array of shape (..., J, 3) joint positions
        bones: List of (start_idx, end_idx) tuples

    Returns:
        lengths: Array of shape (..., len(bones)) bone lengths
    """
    lengths = []
    for start, end in bones:
        bone_vec = joints[..., end, :] - joints[..., start, :]
        length = np.linalg.norm(bone_vec, axis=-1)
        lengths.append(length)
    return np.stack(lengths, axis=-1)
