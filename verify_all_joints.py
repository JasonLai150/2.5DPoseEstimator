import numpy as np
import matplotlib.pyplot as plt

def verify_all():
    print("Running Global Joint Drift Experiment...")
    # Load matrices
    res = np.load('results/result.npz')
    w1, w2 = res['w1'], res['w2']
    
    poses_test = np.load('aggregated_data/poses_test.npy')
    joint_names = list(np.load('aggregated_data/joint_names.npy'))
    
    # Grab the very first pose
    x_sample = poses_test[0:1] 
    x = np.nan_to_num(x_sample, nan=0.0)
    
    # Simulate network identically mapping the matrices explicitly
    latent = np.einsum('bJc,Jj->bjc', x, w1)
    y = np.einsum('bjc,jJ->bJc', latent, w2)
    
    # Project back to 2D
    z_x = x[..., 2:]
    z_x_safe = np.where(np.abs(z_x) < 1e-3, 1.0, z_x)
    z_mean = np.mean(z_x_safe)
    x_proj = (x[..., :2] / z_x_safe) * (z_mean / 1000.0)
    
    z_y = y[..., 2:]
    z_y_safe = np.where(np.abs(z_y) < 1e-3, 1.0, z_y)
    y_proj = (y[..., :2] / z_y_safe) * (z_mean / 1000.0)
    
    gt = x_proj[0]
    pred = y_proj[0]
    
    # Mathematically calculate Euclidean distances of the drift
    dists = np.linalg.norm(gt - pred, axis=1)
    
    # ==========================
    # 1. Bar Chart of Shift Magnitude
    # ==========================
    plt.figure(figsize=(12,6))
    plt.bar(joint_names, dists, color='purple', alpha=0.7)
    plt.ylabel('Euclidean Drift Variance (Normalized Scale)', fontsize=12)
    plt.title('Absolute Drift Variance per Joint', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('results/joint_drift_bar.png', dpi=150)
    
    # ==========================
    # 2. Scatter Plot Displacement Map
    # ==========================
    plt.figure(figsize=(8,8))
    plt.gca().invert_yaxis() # Match physical top-down image coordinates
    
    plt.scatter(gt[:, 0], gt[:, 1], color='red', s=50, label='Ground Truth', zorder=5)
    plt.scatter(pred[:, 0], pred[:, 1], color='blue', s=80, marker='x', label='Predicted Autoencoder', zorder=10)
    
    # Draw vector line connecting each GT to its Prediction to visually trace the exact shift
    for i in range(len(joint_names)):
        plt.plot([gt[i, 0], pred[i, 0]], [gt[i, 1], pred[i, 1]], color='black', alpha=0.3, linewidth=1)
        # Shift label slightly to be readable
        plt.text(gt[i,0] + 0.02, gt[i,1], joint_names[i], fontsize=8, alpha=0.8)
        
    plt.title('2D Vector Displacement Skeleton Map (GT vs Pred)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/joint_drift_scatter.png', dpi=150)
    
    print("Saved macroscopic variance plots!")

if __name__ == '__main__':
    verify_all()
