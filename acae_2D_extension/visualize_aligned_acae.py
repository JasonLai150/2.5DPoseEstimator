import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def identify_dataset(mask, joint_names):
    has_nose = mask[joint_names.index('nose')]
    # In our unified skeleton, Fit3D has 'nose' filled out, H36M does not
    if has_nose:
        return "Fit3D"
    return "Human3.6M"

def visualize_reconstructions(w1, w2, poses_test, joint_names, file_name='acae_data/pose_reconstructions.png'):
    print("Visualizing reconstructions...")
    np.random.seed(42)
    
    mask_test = ~np.isnan(poses_test).all(axis=-1)
    has_nose = mask_test[:, joint_names.index('nose')]
    
    fit3d_idxs = np.where(has_nose)[0]
    h36m_idxs = np.where(~has_nose)[0]
    
    selected = []
    # Grab 2 H36M, 3 Fit3D
    if len(h36m_idxs) >= 2: selected.extend(np.random.choice(h36m_idxs, 2, replace=False))
    if len(fit3d_idxs) >= 3: selected.extend(np.random.choice(fit3d_idxs, 3, replace=False))
    
    sample_idxs = np.array(selected)
    x_sample = poses_test[sample_idxs]
    mask_sample = mask_test[sample_idxs]
    
    x = np.nan_to_num(x_sample, nan=0.0)
    is_valid = mask_sample.astype(np.float32)
    
    # Forward Pass
    w1_exp = w1[np.newaxis, :, :] * is_valid[..., np.newaxis]
    w1_sum = w1_exp.sum(axis=1, keepdims=True) + 1e-9
    w1_norm = w1_exp / w1_sum
    
    latent = np.einsum('bjc,bjJ->bJc', x, w1_norm)
    y = np.einsum('bJc,Jj->bjc', latent, w2)
    
    # Project to 2D
    z_x = x[..., 2:]
    z_y = y[..., 2:]
    z_x_safe = np.where(np.abs(z_x) < 1e-3, 1.0, z_x)
    z_y_safe = np.where(np.abs(z_y) < 1e-3, 1.0, z_y)
    z_mean = np.mean(z_x_safe, axis=1, keepdims=True)
    
    x_proj = (x[..., :2] / z_x_safe) * (z_mean / 1000.0)
    y_proj = (y[..., :2] / z_y_safe) * (z_mean / 1000.0)
    
    # Bones mapping
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
            
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Aligned ACAE Reconstructions', fontsize=16)
    
    for i in range(5):
        if i >= len(sample_idxs): break
        dataset_name = identify_dataset(mask_sample[i], joint_names)
        
        ax1 = axes[0, i]
        ax1.set_title(f"Input GT ({dataset_name})")
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        
        ax2 = axes[1, i]
        ax2.set_title("Autoencoder Output")
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        for j_idx, j_name in enumerate(joint_names):
            if mask_sample[i, j_idx]:
                ax1.scatter(x_proj[i, j_idx, 0], x_proj[i, j_idx, 1], color='red', s=20)
            ax2.scatter(y_proj[i, j_idx, 0], y_proj[i, j_idx, 1], color='orange', s=20)
                
        for p1, p2 in bone_idxs:
            if mask_sample[i, p1] and mask_sample[i, p2]:
                ax1.plot([x_proj[i, p1, 0], x_proj[i, p2, 0]], 
                         [x_proj[i, p1, 1], x_proj[i, p2, 1]], 'b-', linewidth=1.5, alpha=0.5)
            ax2.plot([y_proj[i, p1, 0], y_proj[i, p2, 0]], 
                     [y_proj[i, p1, 1], y_proj[i, p2, 1]], 'g-', linewidth=1.5, alpha=0.5)
                        
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    print(f"Saved reconstructions to '{file_name}'")

def visualize_latent_mapping(w1, joint_names, file_name='acae_data/latent_mapping_heatmap.png'):
    """
    w1 shape is (28, 17) mapping from 28 unified joints to 17 latent points.
    We visualize this as a heatmap.
    """
    print("Visualizing latent mapping heatmap...")
    # The latent points are in Left-Right-Center order: 6 L, 6 R, 5 C.
    # Because of our alignment loss, these 17 points specifically map to:
    lrc_names = [
        'lhip', 'lknee', 'lankle', 'lshoulder', 'lelbow', 'lwrist', # Left 6
        'rhip', 'rknee', 'rankle', 'rshoulder', 'relbow', 'rwrist', # Right 6
        'pelvis', 'spine', 'thorax', 'neck', 'headtop'              # Center 5
    ]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(w1, xticklabels=lrc_names, yticklabels=joint_names, cmap='viridis')
    plt.title("ACAE Encoder Mapping: Unified Joints to 17 Latent Points", fontsize=16)
    plt.xlabel("Latent Point Semantic Identity (Enforced by Alignment Loss)", fontsize=12)
    plt.ylabel("Input Unified Joint", fontsize=12)
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    print(f"Saved latent mapping heatmap to '{file_name}'")

def main():
    print("Loading data...")
    poses_test = np.load('acae_data/poses_test.npy')
    joint_names = list(np.load('acae_data/joint_names.npy'))
    
    if os.path.exists('acae_data/result.npz'):
        res = np.load('acae_data/result.npz')
        w1, w2 = res['w1'], res['w2']
        
        visualize_reconstructions(w1, w2, poses_test, joint_names)
        visualize_latent_mapping(w1, joint_names)
    else:
        print("Could not find acae_data/result.npz. Please finish training first!")

if __name__ == '__main__':
    main()
