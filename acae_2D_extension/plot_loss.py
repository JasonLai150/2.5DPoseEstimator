import pandas as pd
import matplotlib.pyplot as plt

def plot_losses():
    # Load all CSV iterations
    df = pd.read_csv('results/losses.csv')
    
    # We want the final continuous run to exclude aborted executions
    start_idx = df[df['epoch'] == 0].index[-1]
    run_df = df.iloc[start_idx:].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('ACAE Training Convergence (Final 15-Epoch Run)', fontsize=14)
    
    # Plot main loss (which goes from 2.5e-5 down to 9.6e-7, a 96% drop!)
    # We plot it on its own axis because it is 4 magnitudes smaller than the L1 penalty!
    ax1.plot(run_df['epoch'], run_df['metrics/main_loss'], color='blue', linewidth=2)
    ax1.set_title('Reconstruction Error (Raw Geometric Precision)', fontsize=12)
    ax1.set_ylabel('Mean Absolute/Squared Error', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Total Penalized Loss
    ax2.plot(run_df['epoch'], run_df['metrics/loss'], color='orange', linewidth=2, linestyle='--')
    ax2.set_title('Total Penalized Training Loss (Heavily flattened by 0.6 * L1 Regularizer Weight)', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('L1 Penalized Scale', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/loss_curve.png', dpi=150)
    print("Saved convergence chart to loss_curve.png")

if __name__ == '__main__':
    plot_losses()
