import numpy as np

def validate():
    print("Loading test data...")
    poses_test = np.load('aggregated_data/poses_test.npy')
    
    print("Loading ACAE weights from result.npz...")
    try:
        res = np.load('results/result.npz')
        w1 = res['w1']
        w2 = res['w2']
    except FileNotFoundError:
        print("Error: result.npz not found. You must train the model first.")
        return
        
    print(f"Test data shape: {poses_test.shape}")
    
    # Identify valid joints vs missing joints
    mask_test = np.isfinite(poses_test).all(axis=-1)
    
    # Impute 0.0 for missing coordinates so math operations don't yield NaN
    x = np.nan_to_num(poses_test, nan=0.0)
    is_valid = mask_test.astype(np.float32)  # [b, j]
    
    # Dynamically normalize Encoder Weights to ignore missing inputs correctly
    w1_exp = w1[np.newaxis, :, :] * is_valid[..., np.newaxis]
    w1_sum = w1_exp.sum(axis=1, keepdims=True) + 1e-9
    w1_norm = w1_exp / w1_sum
    
    print("Encoding to Latent Space...")
    latent = np.einsum('bjc,bjJ->bJc', x, w1_norm)
    
    print("Decoding to reconstructed 3D pose...")
    y = np.einsum('bJc,Jj->bjc', latent, w2)
    
    # Project the 3D reconstructed coordinates back to the 2D plane correctly
    # matching the `splat` function from acae.py's native focal loss.
    z_x = x[..., 2:]
    z_y = y[..., 2:]
    
    # safeguard division by zero
    z_x_safe = np.where(np.abs(z_x) < 1e-3, 1.0, z_x)
    z_y_safe = np.where(np.abs(z_y) < 1e-3, 1.0, z_y)
    
    # Since we set Z=1000 uniformly in the datasets, z_mean evaluates to 1000.0
    z_mean = np.mean(z_x_safe, axis=1, keepdims=True)
    
    # Scale projections
    x_proj = (x[..., :2] / z_x_safe) * (z_mean / 1000.0)
    y_proj = (y[..., :2] / z_y_safe) * (z_mean / 1000.0)
    
    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE) strictly on valid 2D focal projections
    diff = np.abs(x_proj - y_proj)
    sq_diff = np.square(x_proj - y_proj)
    
    valid_diff = diff[mask_test]
    valid_sq_diff = sq_diff[mask_test]
    
    mae = np.mean(valid_diff)
    mse = np.mean(valid_sq_diff)
    
    print("\n--- Validation Results ---")
    print(f"Evaluated on {np.sum(mask_test)} valid joints across {len(poses_test)} poses.")
    print(f"Mean Absolute Error (MAE): {mae:.6f} (Normalized Scale)")
    print(f"Mean Squared Error (MSE):  {mse:.6f} (Normalized Scale)")
    print("--------------------------")

if __name__ == '__main__':
    validate()
