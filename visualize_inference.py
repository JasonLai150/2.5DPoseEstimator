import os
import sys
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
    VISUALIZATION_CHECKPOINT_PATH,
)

# Add paths to load the required modules
sys.path.append(os.path.abspath('vdp3d'))
sys.path.append(os.path.abspath('acae_2D_extension'))

from common.model import TemporalModel
import importlib.util

ensure_artifact_dirs()

# Dynamically load acae_2.5d_torch.py
acae_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'acae_2D_extension', 'affine_combining_autoencoder', 'acae_2.5d_torch.py'))
spec = importlib.util.spec_from_file_location("acae_torch", acae_path)
acae_module = importlib.util.module_from_spec(spec)
sys.modules["acae_torch"] = acae_module
spec.loader.exec_module(acae_module)
load_acae_from_checkpoint = acae_module.load_acae_from_checkpoint
from prepare_h36m_fit3d import build_unified_skeleton, load_fit3d_3d

# ==========================================================
# 1. Dataset Loading and Preparation
# ==========================================================
def get_sample_fit3d_sequences(tar_path='data/fit3d_train.tar.gz'):
    raw_seqs = load_fit3d_3d(tar_path)
    unified_names, _, body25_to_unified = build_unified_skeleton()
    
    J_unified = len(unified_names)
    processed_seqs = []
    
    for seq in raw_seqs:
        seq = seq[::4] # Subsample to save time
        T = len(seq)
        if T < 243: continue
        
        unified_seq = np.zeros((T, J_unified, 3), dtype=np.float32)
        for b_idx, u_idx in enumerate(body25_to_unified):
            if u_idx >= 0:
                unified_seq[:, u_idx, :] = seq[:, b_idx, :]
                
        pelvis = unified_seq[:, 0:1, :].copy()
        unified_seq = unified_seq - pelvis
        processed_seqs.append(unified_seq)
        
    np.random.seed(42)
    np.random.shuffle(processed_seqs)
    split_idx = int(len(processed_seqs) * 0.9)
    test_seqs = processed_seqs[split_idx:]
    
    return test_seqs[:5], unified_names # Just grab 5 sequences

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
        x_flat = x.reshape(B * T, J, C)
        
        latent_17_2d = self.acae.encode(x_flat).contiguous().reshape(B, T, 17, C).contiguous()
        latent_17_3d = self.vdp3d(latent_17_2d)
        
        latent_17_3d = latent_17_3d.contiguous()
        B_out, T_out, J_out, C_out = latent_17_3d.shape
        latent_17_3d_flat = latent_17_3d.reshape(B_out * T_out, J_out, C_out)
        
        return self.acae.decode(latent_17_3d_flat).contiguous().reshape(B_out, T_out, 28, C_out)

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
    
    # Plot points
    valid_joints = np.abs(pose3d).sum(axis=-1) > 1e-5
    ax.scatter(pose3d[valid_joints, 0], pose3d[valid_joints, 2], -pose3d[valid_joints, 1], c=c, s=15)
    
    # Plot bones
    for b1, b2 in bones:
        if b1 in joint_names and b2 in joint_names:
            idx1 = joint_names.index(b1)
            idx2 = joint_names.index(b2)
            if valid_joints[idx1] and valid_joints[idx2]:
                x_vals = [pose3d[idx1, 0], pose3d[idx2, 0]]
                y_vals = [pose3d[idx1, 2], pose3d[idx2, 2]]
                z_vals = [-pose3d[idx1, 1], -pose3d[idx2, 1]] # Negate Y for visual 'up'
                ax.plot(x_vals, y_vals, z_vals, c=c)
                
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    
    # Set standard limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def main():
    test_seqs, joint_names = get_sample_fit3d_sequences()
    
    # Load Model (We use the fine-tuned one if it exists, otherwise the zero-shot one)
    vdp3d_core = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False)
    model = BridgedTemporalModel(vdp3d_core).cuda()
    
    mixed_ckpt = first_existing_path(MIXED_CHECKPOINT_PATH, 'checkpoint/mixed_finetuned_model.bin')
    if mixed_ckpt.exists():
        print("Visualizing using the Fine-Tuned Model...")
        ckpt = torch.load(mixed_ckpt, map_location='cpu')
    else:
        print("Visualizing using the Zero-Shot Model...")
        ckpt = torch.load(first_existing_path(VISUALIZATION_CHECKPOINT_PATH, 'checkpoint/epoch_110.bin'), map_location='cpu')
        
    model.load_state_dict(ckpt['model_pos'])
    model.eval()
    
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle('VideoPose3D + ACAE Bridge Inference on Fit3D', fontsize=16)
    
    pad = (243 - 1) // 2
    
    for i, seq in enumerate(test_seqs):
        with torch.no_grad():
            seq_padded = np.pad(seq, ((pad, pad), (0, 0), (0, 0)), 'edge')
            inputs_2d_padded = torch.tensor(seq_padded[..., :2], dtype=torch.float32).cuda().unsqueeze(0)
            predicted_3d = model(inputs_2d_padded)
        
        # Pick the middle frame to visualize
        mid_idx = len(seq) // 2
        gt_pose = seq[mid_idx]
        pred_pose = predicted_3d[0, mid_idx].cpu().numpy()
        
        # Plot GT
        ax_gt = fig.add_subplot(2, 5, i + 1, projection='3d')
        plot_skeleton(ax_gt, gt_pose, joint_names, f"Seq {i+1} GT", c='green')
        
        # Plot Pred
        ax_pred = fig.add_subplot(2, 5, i + 6, projection='3d')
        plot_skeleton(ax_pred, pred_pose, joint_names, f"Seq {i+1} Pred", c='red')

    plt.tight_layout()
    plt.savefig('inference_results.png', dpi=300)
    print("\nSaved visualization to inference_results.png!")

if __name__ == '__main__':
    main()
