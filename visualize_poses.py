import numpy as np
import matplotlib.pyplot as plt
import os

def visualize(w1=None, w2=None, epoch_num=None):
    print("Loading test data and model weights...")
    poses_test = np.load('aggregated_data/poses_test.npy')
    joint_names = list(np.load('aggregated_data/joint_names.npy'))
    
    if w1 is None or w2 is None:
        res = np.load('results/result.npz')
        w1, w2 = res['w1'], res['w2']
    
    # Pick 5 random poses that are mostly complete
    np.random.seed(42)
    
    # Safely identify dataset populations mechanically before slicing
    mask_all = ~np.isnan(poses_test).all(axis=2)
    has_spine = mask_all[:, joint_names.index('spine')]
    has_nose = mask_all[:, joint_names.index('nose')]
    
    coco_idxs = np.where(has_nose)[0]
    h36m_idxs = np.where(has_spine)[0]
    mpii_idxs = np.where(~has_nose & ~has_spine)[0]
    
    # Dynamically extract at least 1 random representative array from each dataset pool
    selected = []
    if len(coco_idxs): selected.append(np.random.choice(coco_idxs))
    if len(mpii_idxs): selected.append(np.random.choice(mpii_idxs))
    if len(h36m_idxs): selected.append(np.random.choice(h36m_idxs))
    
    # Backfill safely to hit the 5-column plot quota
    while len(selected) < 5 and len(selected) < len(poses_test):
        candidate = np.random.choice(len(poses_test))
        if candidate not in selected:
            selected.append(candidate)
            
    sample_idxs = np.array(selected)
    x_sample = poses_test[sample_idxs]
    
    # Prepare math for ACAE encoder
    mask_test = np.isfinite(x_sample).all(axis=-1)
    x = np.nan_to_num(x_sample, nan=0.0)
    is_valid = mask_test.astype(np.float32)
    
    w1_exp = w1[np.newaxis, :, :] * is_valid[..., np.newaxis]
    w1_sum = w1_exp.sum(axis=1, keepdims=True) + 1e-9
    w1_norm = w1_exp / w1_sum
    
    # Forward Pass
    latent = np.einsum('bjc,bjJ->bJc', x, w1_norm)
    y = np.einsum('bJc,Jj->bjc', latent, w2)
    
    # Project to 2D exactly like ACAE splat objective
    z_x = x[..., 2:]
    z_y = y[..., 2:]
    
    z_x_safe = np.where(np.abs(z_x) < 1e-3, 1.0, z_x)
    z_y_safe = np.where(np.abs(z_y) < 1e-3, 1.0, z_y)
    z_mean = np.mean(z_x_safe, axis=1, keepdims=True)
    
    x_proj = (x[..., :2] / z_x_safe) * (z_mean / 1000.0)
    y_proj = (y[..., :2] / z_y_safe) * (z_mean / 1000.0)
    
    # Common skeleton bones for unified vocabulary
    bones = [
        ('lshoulder', 'lelbow'), ('lelbow', 'lwrist'),
        ('rshoulder', 'relbow'), ('relbow', 'rwrist'),
        ('lshoulder', 'rshoulder'),
        ('lhip', 'lknee'), ('lknee', 'lankle'),
        ('rhip', 'rknee'), ('rknee', 'rankle'),
        ('lhip', 'rhip'),
        ('lshoulder', 'lhip'), ('rshoulder', 'rhip'),
        ('pelvis', 'thorax'), ('thorax', 'upperneck'), ('upperneck', 'headtop')
    ]
    
    # Build list of index pairs for drawing lines
    bone_idxs = []
    for b1, b2 in bones:
        if b1 in joint_names and b2 in joint_names:
            bone_idxs.append((joint_names.index(b1), joint_names.index(b2)))
            
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('ACAE Autoencoder Reconstructions', fontsize=16)
    
    def identify_dataset(mask):
        has_spine = mask[joint_names.index('spine')]
        has_nose = mask[joint_names.index('nose')]
        if has_nose:
            return "COCO"
        elif has_spine:
            return "Human3.6M"
        return "MPII"
    
    for i in range(5):
        dataset_name = identify_dataset(mask_test[i])
        
        # Ground Truth Wireframes
        ax1 = axes[0, i]
        ax1.set_title(f"Input GT ({dataset_name}) #{sample_idxs[i]}")
        ax1.invert_yaxis()  # Image coordinates have Y going down
        ax1.set_aspect('equal')
        
        # Predicted Wireframes
        ax2 = axes[1, i]
        ax2.set_title("Autoencoder Output (Pred)")
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        for j_idx, j_name in enumerate(joint_names):
            # Visually offset the pelvis text and highlight it uniquely 
            # so it doesn't get buried behind the heavily drawn green skeletal line clusters!
            y_offset = 0.15 if j_name == 'pelvis' else 0.00
            t_color = 'purple' if j_name == 'pelvis' else 'black'
            f_weight = 'bold' if j_name == 'pelvis' else 'normal'
            f_size = 9 if j_name == 'pelvis' else 6
            
            # Ground truth
            if mask_test[i, j_idx]:
                ax1.scatter(x_proj[i, j_idx, 0], x_proj[i, j_idx, 1], color='red', s=20, zorder=5)
                ax1.text(x_proj[i, j_idx, 0], x_proj[i, j_idx, 1] + y_offset, j_name, fontsize=f_size, color=t_color, fontweight=f_weight, alpha=0.9, zorder=10)
            
            # Predict output
            ax2.scatter(y_proj[i, j_idx, 0], y_proj[i, j_idx, 1], color='orange', s=20, zorder=5)
            ax2.text(y_proj[i, j_idx, 0], y_proj[i, j_idx, 1] + y_offset, j_name, fontsize=f_size, color=t_color, fontweight=f_weight, alpha=0.9, zorder=10)
                
        for p1, p2 in bone_idxs:
            # Draw GT only if both joints are valid in input
            if mask_test[i, p1] and mask_test[i, p2]:
                ax1.plot([x_proj[i, p1, 0], x_proj[i, p2, 0]], 
                         [x_proj[i, p1, 1], x_proj[i, p2, 1]], 'b-', linewidth=1.5, alpha=0.5)
            
            # Predict output for all validly reconstructed joints
            ax2.plot([y_proj[i, p1, 0], y_proj[i, p2, 0]], 
                     [y_proj[i, p1, 1], y_proj[i, p2, 1]], 'g-', linewidth=1.5, alpha=0.5)
                        
    plt.tight_layout()
    
    file_name = f'results/pose_reconstructions_epoch_{epoch_num}.png' if epoch_num is not None else 'results/pose_reconstructions.png'
    plt.savefig(file_name, dpi=150)
    print(f"Successfully saved visualizations to '{file_name}'!")

if __name__ == '__main__':
    visualize()
