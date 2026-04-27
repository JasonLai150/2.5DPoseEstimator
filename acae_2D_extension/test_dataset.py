import numpy as np

print("Testing input shapes")

poses_train = np.load('aggregated_data/poses_train.npy')
poses_test = np.load('aggregated_data/poses_test.npy')
joint_names = np.load('aggregated_data/joint_names.npy')

print("Poses train shape", poses_train.shape)
print("Poses test shape", poses_test.shape)
print(joint_names.shape)
print(joint_names)
print("sum of nans poses train", np.isnan(poses_train).sum())
print("sum of nans poses test", np.isnan(poses_test).sum())

bad = np.sum(
    (np.isnan(poses_train).sum(axis=2) != 0) &
    (np.isnan(poses_train).sum(axis=2) != 3)
)

print(poses_train.dtype)
print(poses_test.dtype)

print("partially missing joints:", bad)