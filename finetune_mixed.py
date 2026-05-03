import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from common.h36m_dataset import Human36mDataset
from prepare_h36m_fit3d import build_unified_skeleton, load_fit3d_3d, load_h36m_3d

# ==========================================================
# 1. Dataset Loading and Preparation (Sequential)
# ==========================================================
def get_mixed_train_sequences(fit3d_path='data/fit3d_train.tar.gz', h36m_path='data/data_3d_h36m.npz', sample_stride=2):
    print("Loading datasets for Mixed Fine-tuning...")
    unified_names, h36m_to_unified, body25_to_unified = build_unified_skeleton()
    J_unified = len(unified_names)
    
    # 1. Load Fit3D
    print("-> Processing Fit3D...")
    raw_fit3d = load_fit3d_3d(fit3d_path)
    fit3d_seqs = []
    
    f_norm = 2.0
    c_norm = 0.0

    for seq in raw_fit3d:
        seq = seq[::sample_stride]
        T = len(seq)
        if T < 243: continue
        
        unified_seq_3d = np.zeros((T, J_unified, 3), dtype=np.float32)
        for b_idx, u_idx in enumerate(body25_to_unified):
            if u_idx >= 0:
                unified_seq_3d[:, u_idx, :] = seq[:, b_idx, :]
        
        pelvis = unified_seq_3d[:, 0:1, :].copy()
        unified_seq_3d = unified_seq_3d - pelvis
        
        z = unified_seq_3d[..., 2:3] + 5.0
        pos2d = (unified_seq_3d[..., :2] / z) * f_norm + c_norm
        
        fit3d_seqs.append((pos2d, unified_seq_3d))
        
    np.random.seed(42)
    np.random.shuffle(fit3d_seqs)
    split_idx = int(len(fit3d_seqs) * 0.9)
    fit3d_train = fit3d_seqs[:split_idx]
    
    # 2. Load H36M
    print("-> Processing H36M...")
    kps_file = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
    keypoints_2d = kps_file['positions_2d'].item()
    dataset = Human36mDataset(h36m_path)
    
    h36m_train = []
    for subject in dataset.subjects():
        if subject in ['S9', 'S11']: continue # Keep S9, S11 for testing
        for action in dataset[subject].keys():
            pos3d_world = dataset[subject][action]['positions'].astype(np.float32)
            cams = dataset.cameras()[subject]
            
            for cam_idx in range(len(keypoints_2d[subject][action])):
                pos2d = keypoints_2d[subject][action][cam_idx].astype(np.float32)
                cam = cams[cam_idx]
                
                pos2d[..., :2] = normalize_screen_coordinates(pos2d[..., :2], w=cam['res_w'], h=cam['res_h'])
                pos3d_cam = world_to_camera(pos3d_world, R=cam['orientation'], t=cam['translation'])
                
                T = min(len(pos3d_cam), len(pos2d))
                pos3d = pos3d_cam[:T:sample_stride]
                pos2d = pos2d[:T:sample_stride]
                
                pos3d = pos3d - pos3d[:, :1, :]
                
                pos3d_unified = np.zeros((len(pos3d), J_unified, 3), dtype=np.float32)
                pos2d_unified = np.zeros((len(pos2d), J_unified, 2), dtype=np.float32)
                for h_idx, u_idx in enumerate(h36m_to_unified):
                    pos3d_unified[:, u_idx, :] = pos3d[:, h_idx, :]
                    pos2d_unified[:, u_idx, :] = pos2d[:, h_idx, :]
                    
                h36m_train.append((pos2d_unified, pos3d_unified))

    all_train = fit3d_train + h36m_train
    np.random.shuffle(all_train)
    print(f"Total training sequences: {len(all_train)}")
    return all_train

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

# ==========================================================
# 3. Main Workflow
# ==========================================================
def main():
    train_seqs = get_mixed_train_sequences()
    
    print("\nLoading Pre-trained H36M Model for Fine-tuning...")
    vdp3d_core = TemporalModel(17, 2, 17, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024, dense=False)
    
    # Wrap it FIRST because the checkpoint was saved from the wrapper
    model = BridgedTemporalModel(vdp3d_core).cuda()
    
    ckpt = torch.load(
        first_existing_path(PRETRAINED_H36M_CHECKPOINT_PATH, 'checkpoint/epoch_120.bin'),
        map_location='cpu',
    )
    model.load_state_dict(ckpt['model_pos'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    pad = (243 - 1) // 2
    
    print("\n--- Starting Mixed Fine-tuning ---")
    for epoch in range(1, 6): # 5 epochs
        model.train()
        train_loss = 0
        train_frames = 0
        
        for i, item in enumerate(train_seqs):
            pos2d, pos3d = item
            seq_2d_padded = np.pad(pos2d, ((pad, pad), (0, 0), (0, 0)), 'edge')
            
            inputs_3d = torch.tensor(pos3d, dtype=torch.float32).cuda().unsqueeze(0)
            inputs_2d_padded = torch.tensor(seq_2d_padded, dtype=torch.float32).cuda().unsqueeze(0)
            
            optimizer.zero_grad()
            predicted_3d = model(inputs_2d_padded)
            
            valid_mask = (inputs_3d.abs().sum(dim=-1) > 1e-5).float()
            error = torch.norm(predicted_3d - inputs_3d, dim=-1) * valid_mask
            
            loss = error.sum() / valid_mask.sum()
            loss.backward()
            optimizer.step()
            
            train_loss += error.sum().item()
            train_frames += valid_mask.sum().item()
            
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{len(train_seqs)} sequences...")
                
        avg_train_loss = train_loss / train_frames
        print(f"Epoch {epoch}/5 - Mixed Training Loss: {avg_train_loss * 1000:.2f} mm")
        
    print("\nSaving fine-tuned checkpoint...")
    torch.save(
        {
            'epoch': 5,
            'model_pos': model.state_dict(),
            'model_traj': None,
            'optimizer': None,
            'lr': 0.0001,
        },
        MIXED_CHECKPOINT_PATH,
    )
    print(f"Saved to {MIXED_CHECKPOINT_PATH}")
    print("Done! You can now run eval_fit3d.py again using this new checkpoint to compare.")

if __name__ == '__main__':
    main()
