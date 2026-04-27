import json
import zipfile
import scipy.io
import numpy as np
import os


def main():
    test_coco('data/coco_wholebody_train_v1.0.json')

def test_coco(json_path):
    """
    2d dataset
    """
    print("Loading COCO WholeBody...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keys = data.keys()
    print(keys)

    print(len(data['images']))
    print(len(data['annotations']))
    print(len(data['categories']))


    print("Images")
    #print(data['images'][0])
    print(data['images'][0].keys())
    print("Annotations")
    for k,v in data['annotations'][0].items():
        print(k)
        print(v)
    #print("Categories")
    #print(data['categories'][0])
        
    # Standard COCO 17 joints
    coco_names = ['nose', 'leye', 'reye', 'lear', 'rear', 
                  'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 
                  'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle']
                  
    # For normalization, we can use middle of hips as center, and shoulder distance as scale
    # lhip=11, rhip=12, lshoulder=5, rshoulder=6

    """
                  
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

        """
                        
    #return np.array(poses), coco_names

if __name__ == '__main__':
    main()