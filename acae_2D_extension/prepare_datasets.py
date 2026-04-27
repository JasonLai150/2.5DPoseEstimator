import json
import zipfile
import scipy.io
import numpy as np
import os

def normalize_keypoints(joints_2d, mask, center_idx, scale_idx1, scale_idx2):
    # joints_2d: [N, J, 2]
    # mask: [N, J] bool, True if valid (not NaN)
    # This function operates on one example at a time, or broadly
    N, J, _ = joints_2d.shape
    out = np.full_like(joints_2d, np.nan)
    
    for i in range(N):
        if not mask[i, center_idx]:
            continue
            
        center = joints_2d[i, center_idx]
        
        if mask[i, scale_idx1] and mask[i, scale_idx2]:
            scale = np.linalg.norm(joints_2d[i, scale_idx1] - joints_2d[i, scale_idx2])
        else:
            # Fallback if scale joints are missing, just use scale 1 or skip?
            scale = 1.0
            
        if scale < 1e-6:
            scale = 1.0
            
        out[i] = (joints_2d[i] - center) / scale
        
    return out

def load_coco(json_path):
    print("Loading COCO WholeBody...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Standard COCO 17 joints
    coco_names = ['nose', 'leye', 'reye', 'lear', 'rear', 
                  'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 
                  'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle']
                  
    # For normalization, we can use middle of hips as center, and shoulder distance as scale
    # lhip=11, rhip=12, lshoulder=5, rshoulder=6
                  
    poses = []
    for ann in data['annotations']:
        if 'keypoints' in ann:
            kpts = ann['keypoints']
            if len(kpts) >= 51:
                # [x, y, v]
                kpts = np.array(kpts[:51]).reshape(-1, 3)
                # Keep only those with v > 0
                joints = np.full((17, 2), np.nan)
                valid = kpts[:, 2] > 0
                joints[valid] = kpts[valid, :2]
                
                # Check if hips and shoulders exist to normalize
                if valid[11] and valid[12]:
                    center = (joints[11] + joints[12]) / 2.0
                    if valid[5] and valid[6]:
                        scale = np.linalg.norm(joints[5] - joints[6])
                    else:
                        scale = 1.0
                    if scale > 1e-6:
                        joints = (joints - center) / scale
                        # add dummy Z
                        joints_3d = np.concatenate([joints, np.full((17, 1), 1000.0)], axis=-1)
                        poses.append(joints_3d)
                        
    return np.array(poses), coco_names  #(N, 17, 3)

def load_mpii(zip_path):
    print("Loading MPII...")
    # Read mat from zip
    with zipfile.ZipFile(zip_path) as z:
        with z.open('mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat') as f:
            mat = scipy.io.loadmat(f)
            
    mpii_names = ['rankle', 'rknee', 'rhip', 'lhip', 'lknee', 'lankle', 
                  'pelvis', 'thorax', 'upperneck', 'headtop', 
                  'rwrist', 'relbow', 'rshoulder', 'lshoulder', 'lelbow', 'lwrist']
             
    # Extract annotations
    annolist = mat['RELEASE']['annolist'][0,0][0]
    poses = []
    
    for i in range(len(annolist)):
        annorect = annolist[i]['annorect']
        if annorect.size == 0:
            continue
        
        for j in range(annorect.shape[1]):
            rect = annorect[0, j]
            if rect is None or not hasattr(rect, 'dtype') or 'annopoints' not in rect.dtype.names or rect['annopoints'].size == 0:
                continue
                
            points = rect['annopoints'][0,0]['point'][0]
            
            joints = np.full((16, 2), np.nan)
            valid = np.zeros(16, dtype=bool)
            
            for p in points:
                ident = p['id'][0,0]
                if ident < 16:
                    x = p['x'][0,0]
                    y = p['y'][0,0]
                    joints[ident] = [x, y]
                    valid[ident] = True
                    
            if valid[6]: # pelvis is center
                center = joints[6]
                if valid[13] and valid[12]: # lshoulder and rshoulder for scale
                    scale = np.linalg.norm(joints[13] - joints[12])
                else:
                    scale = 1.0
                    
                if scale > 1e-6:
                    joints = (joints - center) / scale
                    joints_3d = np.concatenate([joints, np.full((16, 1), 1000.0)], axis=-1)
                    poses.append(joints_3d)
                    
    return np.array(poses), mpii_names

def load_h36m(npz_path):
    print("Loading Human3.6M...")
    d2 = np.load(npz_path, allow_pickle=True)
    pos_2d = d2['positions_2d'].item()
    
    # H36M usually has 17 joints, similar to COCO but different order
    h36m_names = ['pelvis', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 
                  'spine', 'thorax', 'upperneck', 'headtop', 
                  'lshoulder', 'lelbow', 'lwrist', 'rshoulder', 'relbow', 'rwrist']
                  
    poses = []
    
    for subject in pos_2d.keys():
        for action in pos_2d[subject].keys():
            for cam_idx in range(len(pos_2d[subject][action])):
                # Extract frames for this camera sequence
                frames = pos_2d[subject][action][cam_idx] # Shape: [N_frames, 17, 2]
                
                for i in range(len(frames)):
                    joints = frames[i]
                    
                    # Normalize scale based on pelvis (0), lshoulder(11), rshoulder(14)
                    center = joints[0]
                    scale = np.linalg.norm(joints[11] - joints[14])
                    
                    if scale > 1e-6:
                        norm_joints = (joints - center) / scale
                        # add dummy Z
                        joints_3d = np.concatenate([norm_joints, np.full((17, 1), 1000.0)], axis=-1)
                        poses.append(joints_3d)
                        
    return np.array(poses), h36m_names

def fix_partial_joints(poses):
    # count NaNs per joint
    nan_count = np.isnan(poses).sum(axis=2)

    # joints with some but not all coordinates missing
    partial_mask = (nan_count > 0) & (nan_count < 3)

    # set them to fully missing
    poses[partial_mask, :] = np.nan

    return poses


def main():
    coco_json = 'data/coco_wholebody_train_v1.0.json'
    mpii_zip = 'data/mpii_human_pose_v1_u12_2.zip'
    h36m_npz = 'data/h36m/data/data_2d_h36m_gt.npz'
    
    coco_poses, coco_names = load_coco(coco_json)
    mpii_poses, mpii_names = load_mpii(mpii_zip)
    h36m_poses, h36m_names = load_h36m(h36m_npz)
    
    print(f"COCO poses: {coco_poses.shape}")
    print(f"MPII poses: {mpii_poses.shape}")
    print(f"H36M poses: {h36m_poses.shape}")
    
    # Unified names
    all_names = coco_names + [n for n in mpii_names if n not in coco_names]
    all_names = all_names + [n for n in h36m_names if n not in all_names]
    
    N_coco = len(coco_poses)
    N_mpii = len(mpii_poses)
    N_h36m = len(h36m_poses)
    J = len(all_names)
    
    # Create unified matrices
    poses_train = np.full((N_coco + N_mpii + N_h36m, J, 3), np.nan)
    
    for i, name in enumerate(coco_names):
        j_idx = all_names.index(name)
        poses_train[:N_coco, j_idx, :] = coco_poses[:, i, :]
        
    for i, name in enumerate(mpii_names):
        j_idx = all_names.index(name)
        poses_train[N_coco:N_coco+N_mpii, j_idx, :] = mpii_poses[:, i, :]
        
    for i, name in enumerate(h36m_names):
        j_idx = all_names.index(name)
        poses_train[N_coco+N_mpii:, j_idx, :] = h36m_poses[:, i, :]
        
    print(f"Combined poses: {poses_train.shape}")
    
    # Optional split into train/test (e.g. 90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(poses_train))
    split = int(0.9 * len(poses_train))
    
    train_idx = indices[:split]
    test_idx = indices[split:]
    
    poses_train_final = poses_train[train_idx]
    poses_test_final = poses_train[test_idx]

    poses_train_final = fix_partial_joints(poses_train_final).astype(np.float32)
    poses_test_final = fix_partial_joints(poses_test_final).astype(np.float32)
    
    np.save('aggregated_data/poses_train.npy', poses_train_final)
    np.save('aggregated_data/poses_test.npy', poses_test_final)
    np.save('aggregated_data/joint_names.npy', np.array(all_names))
    
    print("Saved poses_train.npy, poses_test.npy, joint_names.npy")

if __name__ == '__main__':
    main()
