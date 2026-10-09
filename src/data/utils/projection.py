""" Skeleton Projection """
import numpy as np

def generate_angle_noise(degrees,noise):
    deg = np.random.uniform(-noise, noise)
    return degrees + deg

def get_rotation_matrix(axis, degrees, noise=None):
    # axis that gets the intended degrees
    target_axis = axis.lower()

    angles = {"x": 0.0, "y": 0.0, "z": 0.0}
    angles[target_axis] = degrees

    # add random noise to all axes
    if noise:
        for ax in angles:
            rand_noise = np.random.uniform(-noise, noise)
            if ax == target_axis:
                angles[ax] += rand_noise   
            else:
                angles[ax] = rand_noise    

    rx, ry, rz = np.deg2rad([angles["x"], angles["y"], angles["z"]])

    Rx = np.array([[1, 0, 0],[0, np.cos(rx), -np.sin(rx)],[0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],[0, 1, 0],[-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],[np.sin(rz),  np.cos(rz), 0],[0, 0, 1]])

    R = Rz @ Ry @ Rx
    return R

def project_2d(skeleton_3d, K, extrinsics, joint_noise=None, img_height=1080):
    """ Project skeleton [T,J,3] → [T,J,2] 
    
    Note: Standard pinhole projection produces Y-up coordinates (Cartesian).
    We flip to Y-down (image convention) to match MMPose format.
    """
    T, J, _ = skeleton_3d.shape
    projected_2d = np.zeros((T,J,2))

    for i in range(T):
        pts_3d = skeleton_3d[i]                           # [J,3]
        pts_3d_h = np.hstack((pts_3d, np.ones((J,1))))    # [J,4]
        pts_cam = (extrinsics @ pts_3d_h.T).T             # [J,3]
        pts_proj = (K @ pts_cam.T).T                      # [J,3]
        pts_proj_2d = pts_proj[:,:2] / pts_proj[:,2:3]    # [J,2]
        projected_2d[i] = pts_proj_2d

        if joint_noise:
            # body height in pixels (head–root distance)
            height_2d = 2*np.linalg.norm(pts_proj_2d[10] - pts_proj_2d[0]) # head is idx 10 and root is idx 0

            # add Gaussian noise proportional to height
            noise = np.random.randn(J, 2) * (joint_noise * height_2d)
            pts_proj_2d += noise

        # Flip Y to image convention (Y-down) to match MMPose format
        pts_proj_2d[:, 1] = img_height - pts_proj_2d[:, 1]
        
        projected_2d[i] = pts_proj_2d

    return projected_2d   # [T,J,2]