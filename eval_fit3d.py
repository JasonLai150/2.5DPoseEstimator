import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from run_paths import (
    ensure_artifact_dirs,
    first_existing_path,
    ACAE_CHECKPOINT_PATH,
    MIXED_CHECKPOINT_PATH,
    PRETRAINED_H36M_CHECKPOINT_PATH,
)

# Add paths to load the required modules
sys.path.append(os.path.abspath('vdp3d'))
sys.path.append(os.path.abspath('acae_2D_extension'))

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates, world_to_camera
import importlib.util

ensure_artifact_dirs()

# Dynamically load acae_2.5d_torch.py to bypass tensorflow dependency in __init__.py
acae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'acae_2D_extension', 'affine_combining_autoencoder', 'acae_2.5d_torch.py'))
spec = importlib.util.spec_from_file_location("acae_torch", acae_path)
acae_module = importlib.util.module_from_spec(spec)
sys.modules["acae_torch"] = acae_module
spec.loader.exec_module(acae_module)
load_acae_from_checkpoint = acae_module.load_acae_from_checkpoint
from prepare_h36m_fit3d import build_unified_skeleton, load_fit3d_3d, load_h36m_3d

# ==========================================================
# 1. Dataset Loading and Preparation (Sequential)
# ==========================================================
def get_fit3d_test_sequences(tar_path='data/fit3d_train.tar.gz', sample_stride=2):
    print("Loading Fit3D sequences for Evaluation...")
    raw_seqs = load_fit3d_3d(tar_path)
    unified_names, h36m_to_unified, body25_to_unified = build_unified_skeleton()
    
    J_unified = len(unified_names)
    processed_seqs = []
    
    # "Standard" camera intrinsics for Fit3D if not found (based on typical datasets)
    f_norm = 2.0
    c_norm = 0.0

    for seq in raw_seqs:
        seq = seq[::sample_stride]
        T = len(seq)
        if T < 243: continue
        
        unified_seq_3d = np.zeros((T, J_unified, 3), dtype=np.float32)
        for b_idx, u_idx in enumerate(body25_to_unified):
            if u_idx >= 0:
                unified_seq_3d[:, u_idx, :] = seq[:, b_idx, :]

        # Root-centering
        pelvis = unified_seq_3d[:, 0:1, :].copy()
        unified_seq_3d = unified_seq_3d - pelvis
        
        # Simple perspective projection (mocked since real intrinsics are hard to extract)
        z = unified_seq_3d[..., 2:3] + 5.0 
        pos2d = (unified_seq_3d[..., :2] / z) * f_norm + c_norm

        processed_seqs.append((pos2d, unified_seq_3d))
        
    # We take the exact same 10% test split as before for consistency
    np.random.seed(42)
    np.random.shuffle(processed_seqs)
    split_idx = int(len(processed_seqs) * 0.9)
    test_seqs = processed_seqs[split_idx:]
    
    print(f"Loaded {len(test_seqs)} valid Fit3D test sequences.")
    return test_seqs, unified_names

def get_h36m_test_sequences():
    sys.path.insert(0, os.path.abspath('vdp3d'))
    from common.h36m_dataset import Human36mDataset
    from common.camera import normalize_screen_coordinates

    dataset = Human36mDataset('data/data_3d_h36m.npz')
    unified_names, h36m_to_unified, _ = build_unified_skeleton()
    J_unified = len(unified_names)

    # Load the exact same 2D keypoints that VDP3D was trained/evaluated on
    kps_file = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
    keypoints_2d = kps_file['positions_2d'].item()

    # Apply the same screen normalization as run.py line 148
    for subject in keypoints_2d.keys():
        for action in keypoints_2d[subject].keys():
            for cam_idx, kps in enumerate(keypoints_2d[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints_2d[subject][action][cam_idx] = kps

    test_seqs_bridged = []
    test_seqs_native = []

    for subject in ['S9', 'S11']:
        if subject not in dataset.subjects(): continue
        for action in dataset[subject].keys():
            # 3D ground truth in world space (T, 17, 3)
            pos3d_world = dataset[subject][action]['positions'].astype(np.float32)
            cams = dataset.cameras()[subject]

            for cam_idx in range(len(keypoints_2d[subject][action])):
                pos2d = keypoints_2d[subject][action][cam_idx].astype(np.float32)  # (T, 17, 2)
                cam = cams[cam_idx]

                # Transform to camera space
                pos3d_cam = world_to_camera(pos3d_world, R=cam['orientation'], t=cam['translation'])

                # Align lengths and subsample
                T = min(len(pos3d_cam), len(pos2d))
                pos3d = pos3d_cam[:T:4]
                pos2d  = pos2d[:T:4]
                T = len(pos3d)
                if T < 243: continue

                # Root-center 3D targets to match VDP3D relative pose convention
                pos3d = pos3d - pos3d[:, :1, :]

                # 1. Native: proper perspective 2D + root-centered 3D (17 joints)
                test_seqs_native.append((pos2d, pos3d))

                # 2. Bridged: expand to 28-joint unified skeleton
                pos3d_unified = np.zeros((T, J_unified, 3), dtype=np.float32)
                pos2d_unified = np.zeros((T, J_unified, 2), dtype=np.float32)
                for h_idx, u_idx in enumerate(h36m_to_unified):
                    pos3d_unified[:, u_idx, :] = pos3d[:, h_idx, :]
                    pos2d_unified[:, u_idx, :] = pos2d[:, h_idx, :]

                test_seqs_bridged.append((pos2d_unified, pos3d_unified))

    print(f"Loaded {len(test_seqs_native)} valid H36M test sequences.")
    return test_seqs_bridged, test_seqs_native


# ==========================================================
# 2. Model Definition
# ==========================================================
class BridgedTemporalModel(nn.Module):
    def __init__(self, vdp3d_model):
        super().__init__()
        ckpt_path = first_existing_path(ACAE_CHECKPOINT_PATH, 'acae_data/checkpoints/acae_aligned_checkpoint.pth')
        self.acae = load_acae_from_checkpoint(str(ckpt_path), device='cuda' if torch.cuda.is_available() else 'cpu', freeze=True)
        self.vdp3d = vdp3d_model

    def forward(self, x):
        x = x.contiguous()
        B, T, J, C = x.shape
        
        # Bypass bridge if input is already 17 joints (Native VideoPose3D inference)
        if J == 17:
            return self.vdp3d(x)
            
        x_flat = x.reshape(B * T, J, C)
        
        latent_17_2d = self.acae.encode(x_flat).contiguous().reshape(B, T, 17, C).contiguous()
        latent_17_3d = self.vdp3d(latent_17_2d)
        
        latent_17_3d = latent_17_3d.contiguous()
        B_out, T_out, J_out, C_out = latent_17_3d.shape
        latent_17_3d_flat = latent_17_3d.reshape(B_out * T_out, J_out, C_out)
        
        return self.acae.decode(latent_17_3d_flat).contiguous().reshape(B_out, T_out, 28, C_out)

    def receptive_field(self):
        return self.vdp3d.receptive_field()

def evaluate(model, test_seqs, device):
    model.eval()
    total_error = 0
    total_frames = 0
    pad = (model.receptive_field() - 1) // 2

    with torch.no_grad():
        for item in test_seqs:
            # Sequences can be either a plain 3D array (Fit3D) or a (pos2d, pos3d) tuple (H36M)
            pos2d, pos3d = item
            seq_2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), 'edge')

            inputs_3d = torch.tensor(pos3d, dtype=torch.float32, device=device).unsqueeze(0)
            inputs_2d_padded = torch.tensor(seq_2d_padded, dtype=torch.float32, device=device).unsqueeze(0)

            predicted_3d = model(inputs_2d_padded)

            # Mask out invalid joints and ensure root is handled correctly (both are relative to pelvis)
            valid_mask = (inputs_3d.abs().sum(dim=-1) > 1e-5).float()
            error = torch.norm(predicted_3d - inputs_3d, dim=-1) * valid_mask

            total_error += error.sum().item()
            total_frames += valid_mask.sum().item()

    return total_error / total_frames if total_frames > 0 else float('inf')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        default=str(MIXED_CHECKPOINT_PATH),
        help='Wrapper checkpoint to evaluate for Fit3D and manual bridged/native H36M checks.',
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device for manual evaluation.',
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip eval_visualizations.png generation.',
    )
    parser.add_argument(
        '--skip-fit3d',
        action='store_true',
        help='Skip Fit3D loading/evaluation for faster H36M-only debugging.',
    )
    parser.add_argument(
        '--skip-h36m',
        action='store_true',
        help='Skip H36M loading/evaluation for Fit3D-only debugging.',
    )
    return parser.parse_args()

def plot_skeleton(ax, pose3d, joint_names, title, c='blue'):
    bones = [
        ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
        ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
        ('LShoulder', 'RShoulder'),
        ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
        ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
        ('LHip', 'RHip'),
        ('LShoulder', 'LHip'), ('RShoulder', 'RHip'),
        ('Pelvis', 'Thorax'), ('Thorax', 'Neck'), ('Neck', 'Head')
    ]
    
    valid_joints = np.abs(pose3d).sum(axis=-1) > 1e-5
    ax.scatter(pose3d[valid_joints, 0], pose3d[valid_joints, 2], -pose3d[valid_joints, 1], c=c, s=15)
    
    for b1, b2 in bones:
        if b1 in joint_names and b2 in joint_names:
            idx1 = joint_names.index(b1)
            idx2 = joint_names.index(b2)
            if valid_joints[idx1] and valid_joints[idx2]:
                ax.plot([pose3d[idx1, 0], pose3d[idx2, 0]], 
                        [pose3d[idx1, 2], pose3d[idx2, 2]], 
                        [-pose3d[idx1, 1], -pose3d[idx2, 1]], c=c)
                
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no CUDA device is available.')
    device = torch.device(args.device)

    test_seqs_fit3d = None
    joint_names = None
    test_seqs_h36m_bridged = None
    test_seqs_h36m_native = None

    if not args.skip_fit3d:
        test_seqs_fit3d, joint_names = get_fit3d_test_sequences()
    if not args.skip_h36m:
        test_seqs_h36m_bridged, test_seqs_h36m_native = get_h36m_test_sequences()
    
    print(f"\nLoading wrapper checkpoint: {args.checkpoint}")
    vdp3d_core = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False)
    
    # Wrap it FIRST because the checkpoint was saved from the wrapper (keys have 'vdp3d.' prefix)
    model = BridgedTemporalModel(vdp3d_core).to(device)
    
    # Load the trained model
    ckpt_path = first_existing_path(args.checkpoint, MIXED_CHECKPOINT_PATH, PRETRAINED_H36M_CHECKPOINT_PATH)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_pos'])
    
    loss_fit3d = None
    loss_h36m_bridged = None
    loss_h36m_native = None

    if test_seqs_fit3d is not None:
        print("\n--- Running Inference on Fit3D Test Set ---")
        print("Inputs: mocked/projected Fit3D 2D from 3D (debug only, not a valid deployment metric)")
        loss_fit3d = evaluate(model, test_seqs_fit3d, device)
        print(f"Fit3D Evaluation MPJPE Loss: {loss_fit3d * 1000:.2f} mm")

    if test_seqs_h36m_bridged is not None:
        print("\n--- Running Inference on H36M Test Set (WITH ACAE Bridge) ---")
        print("Inputs: official H36M 2D keypoints expanded to unified skeleton")
        loss_h36m_bridged = evaluate(model, test_seqs_h36m_bridged, device)
        print(f"H36M Evaluation (WITH Bridge) MPJPE Loss: {loss_h36m_bridged * 1000:.2f} mm")
        
        print("\n--- Running Inference on H36M Test Set (SAME CHECKPOINT, BRIDGE BYPASSED) ---")
        print("Inputs: official H36M 17-joint 2D keypoints; this is not the pretrained VideoPose3D baseline.")
        loss_h36m_native = evaluate(model, test_seqs_h36m_native, device)
        print(f"H36M Evaluation (Bridge Bypassed) MPJPE Loss: {loss_h36m_native * 1000:.2f} mm")

    if args.skip_visualization:
        return

    if test_seqs_fit3d is None:
        print("\nSkipping visualizations because Fit3D evaluation was skipped.")
        return

    print("\n--- Generating Visualizations ---")
    fig = plt.figure(figsize=(20, 8))
    title_parts = [f'Fit3D: {loss_fit3d * 1000:.0f}mm']
    if loss_h36m_bridged is not None:
        title_parts.append(f'H36M Bridged: {loss_h36m_bridged * 1000:.0f}mm')
        title_parts.append(f'H36M Bypassed: {loss_h36m_native * 1000:.0f}mm')
    fig.suptitle('VideoPose3D + ACAE Bridge (' + ' | '.join(title_parts) + ')', fontsize=16)
    
    pad = (243 - 1) // 2
    for i, item in enumerate(test_seqs_fit3d[:5]): # Visualize first 5 sequences
        pos2d, gt_pose_seq = item
        with torch.no_grad():
            seq_2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), 'edge')
            inputs_2d_padded = torch.tensor(seq_2d_padded, dtype=torch.float32, device=device).unsqueeze(0)
            predicted_3d = model(inputs_2d_padded)
        
        mid_idx = len(gt_pose_seq) // 2
        gt_pose = gt_pose_seq[mid_idx]
        pred_pose = predicted_3d[0, mid_idx].cpu().numpy()
        
        ax_gt = fig.add_subplot(2, 5, i + 1, projection='3d')
        plot_skeleton(ax_gt, gt_pose, joint_names, f"Seq {i+1} GT", c='green')
        
        ax_pred = fig.add_subplot(2, 5, i + 6, projection='3d')
        plot_skeleton(ax_pred, pred_pose, joint_names, f"Seq {i+1} Pred", c='red')

    plt.tight_layout()
    plt.savefig('eval_visualizations.png', dpi=300)
    print("Saved evaluation visualizations to eval_visualizations.png!")

if __name__ == '__main__':
    main()
