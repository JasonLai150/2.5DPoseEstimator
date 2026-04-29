import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

def identify_dataset(mask, joint_names):
    has_nose = mask[joint_names.index('nose')]
    if has_nose:
        return "Fit3D"
    return "Human3.6M"

def visualize_reconstructions(w1, w2, poses_test, joint_names, file_name='acae_data/pose_reconstructions.png'):
    print("Visualizing reconstructions...")
    np.random.seed(42)
    
    # 0.0 is our sentinel for missing joints
    mask_test = (poses_test != 0.0).any(axis=-1)
    has_nose = mask_test[:, joint_names.index('nose')]
    
    fit3d_idxs = np.where(has_nose)[0]
    h36m_idxs = np.where(~has_nose)[0]
    
    print(f"Found {len(fit3d_idxs)} Fit3D and {len(h36m_idxs)} H36M frames.")
    
    selected = []
    # Force 2 H36M and 3 Fit3D if available
    if len(h36m_idxs) >= 2:
        selected.extend(np.random.choice(h36m_idxs, 2, replace=False))
    else:
        selected.extend(h36m_idxs)
        
    if len(fit3d_idxs) >= (5 - len(selected)):
        selected.extend(np.random.choice(fit3d_idxs, 5 - len(selected), replace=False))
    else:
        selected.extend(fit3d_idxs)
    
    sample_idxs = np.array(selected)
    print(f"Selected indices: {sample_idxs}")
    
    x = poses_test[sample_idxs]
    mask_sample = mask_test[sample_idxs]
    
    # Forward Pass math (matches model's missing-joint handling)
    # x: (B, J, 3), w1: (J, L), w2: (L, J)
    is_valid = mask_sample.astype(np.float32)
    w1_exp = w1[np.newaxis, :, :] * is_valid[..., np.newaxis]
    w1_sum = w1_exp.sum(axis=1, keepdims=True) + 1e-9
    w1_norm = w1_exp / w1_sum
    
    latent = np.einsum('bjc,bjJ->bJc', x, w1_norm)
    y = np.einsum('bJc,Jj->bjc', latent, w2)
    
    # --- Centering for Visualization ---
    # Centering on Pelvis (joint 0) makes the "human" shape obvious 
    # and removes camera-offset distortions/skewing.
    pelvis_x = x[:, 0:1, :]
    pelvis_y = y[:, 0:1, :]
    
    x_centered = x - pelvis_x
    y_centered = y - pelvis_y
    
    # In camera coordinates, X is right, Y is down.
    x_proj = x_centered[..., :2]
    y_proj = y_centered[..., :2]
    
    # Bone connections for unified skeleton
    bones = [
        ('lshoulder', 'lelbow'), ('lelbow', 'lwrist'),
        ('rshoulder', 'relbow'), ('relbow', 'rwrist'),
        ('lshoulder', 'rshoulder'),
        ('lhip', 'lknee'), ('lknee', 'lankle'),
        ('rhip', 'rknee'), ('rknee', 'rankle'),
        ('lhip', 'rhip'),
        ('lshoulder', 'lhip'), ('rshoulder', 'rhip'),
        ('pelvis', 'thorax'), ('thorax', 'neck'), ('neck', 'headtop')
    ]
    
    bone_idxs = []
    for b1, b2 in bones:
        if b1 in joint_names and b2 in joint_names:
            bone_idxs.append((joint_names.index(b1), joint_names.index(b2)))
            
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('ACAE Reconstructions (Centered on Pelvis)', fontsize=16)
    
    for i in range(5):
        ax1 = axes[0, i]
        ax2 = axes[1, i]
        
        if i >= len(sample_idxs):
            ax1.set_title("No Data")
            ax2.set_title("No Data")
            continue
            
        dataset_name = identify_dataset(mask_sample[i], joint_names)
        
        ax1.set_title(f"Input {dataset_name} (GT)")
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        
        ax2.set_title(f"ACAE Output (Pred)")
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        # Determine plot limits to keep scale consistent (limit to human-sized range)
        # We clip the view to roughly 1 meter around the centered pelvis
        ax1.set_xlim(-900, 900)
        ax1.set_ylim(1200, -1200) # inverted
        ax2.set_xlim(-900, 900)
        ax2.set_ylim(1200, -1200)
        
        # Plot joints
        for j_idx, j_name in enumerate(joint_names):
            # Only plot joints that were present in the input to avoid "hallucinated" 
            # joints at the origin skewing the visualization
            if mask_sample[i, j_idx]:
                ax1.scatter(x_proj[i, j_idx, 0], x_proj[i, j_idx, 1], color='red', s=20, zorder=5)
                ax2.scatter(y_proj[i, j_idx, 0], y_proj[i, j_idx, 1], color='orange', s=20, zorder=5)
                
        # Plot bones
        for p1, p2 in bone_idxs:
            # Only plot bones if both joints were valid in the input
            if mask_sample[i, p1] and mask_sample[i, p2]:
                ax1.plot([x_proj[i, p1, 0], x_proj[i, p2, 0]], 
                         [x_proj[i, p1, 1], x_proj[i, p2, 1]], 'b-', linewidth=1.5, alpha=0.5)
                ax2.plot([y_proj[i, p1, 0], y_proj[i, p2, 0]], 
                         [y_proj[i, p1, 1], y_proj[i, p2, 1]], 'g-', linewidth=1.5, alpha=0.5)
                        
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    print(f"Saved reconstructions to '{file_name}'")

def visualize_latent_mapping(w1, joint_names, file_name='acae_data/latent_mapping_heatmap.png'):
    print("Visualizing latent mapping heatmap...")
    lrc_names = [
        'lhip', 'lknee', 'lankle', 'lshoulder', 'lelbow', 'lwrist',
        'rhip', 'rknee', 'rankle', 'rshoulder', 'relbow', 'rwrist',
        'pelvis', 'spine', 'thorax', 'neck', 'headtop'
    ]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(w1, xticklabels=lrc_names, yticklabels=joint_names, cmap='viridis')
    plt.title("ACAE Encoder Mapping: Unified Joints to 17 Latent Points", fontsize=16)
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    print(f"Saved latent mapping heatmap to '{file_name}'")

def main():
    print("Loading data and checkpoint...")
    poses_test = np.load('acae_data/poses_test.npy')
    joint_names = list(np.load('acae_data/joint_names.npy'))
    ckpt_path = 'acae_data/checkpoints/acae_aligned_checkpoint.pth'
    
    if os.path.exists(ckpt_path):
        # Load w1, w2 directly from checkpoint to ensure we use the best weights
        ckpt = torch.load(ckpt_path, map_location='cpu')
        w1 = ckpt['w1']
        w2 = ckpt['w2']
        
        visualize_reconstructions(w1, w2, poses_test, joint_names)
        visualize_latent_mapping(w1, joint_names)
    else:
        print(f"Could not find {ckpt_path}. Please finish training first!")

if __name__ == '__main__':
    main()
