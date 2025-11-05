#----------------------------------------------------------------------------------------------------
# Work done at the Intelligent Robotics and Vision Lab, University of Texas at Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Sai Haneesh Allu (2025).
#----------------------------------------------------------------------------------------------------
import yaml
import sys
import numpy as np
sys.path.insert(0,"..")
import cv2
import matplotlib.pyplot as plt

def load_yaml(file_path):
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=yaml.Loader)
    else:
        yaml_params = file_path
    return yaml_params



def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) 
    return xyz_img

def compute_object_mean_position(depth_img, mask, RT_camera, cam_k):
    [fx,fy,px,py] = cam_k
    depth_array = depth_img * (mask / 255)
    mask1 = np.isnan(depth_array)
    depth_array[mask1] = 0.0
    xyz_array = compute_xyz(
        depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
    )
    xyz_array = xyz_array.reshape((-1, 3))

    mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
    xyz_array = xyz_array[mask]
    print(f"mean pose cam link {np.mean(xyz_array, axis=0)}")

    xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
    xyz_base += RT_camera[:3, 3]
    print(f"mean pose base link {np.mean(xyz_base, axis=0)}")

    mean_pose = np.mean(xyz_base, axis=0)
    return mean_pose


def find_point_at_distance(mean_position, target_distance=2.0):
    distance = np.linalg.norm(mean_position)
    
    # Compute the point at (distance - target_distance) along the vector from baselink
    unit_vector = mean_position / distance
    point_distance = distance - target_distance
    point = unit_vector * point_distance
    
    return point


def compute_object_pc_mean(depth_img, mask, RT_camera, cam_k, distance_threshold=2.0):
    # erode the mask to avoid background depths
    mask = cv2.erode(mask, None, 2)
    mean_position = compute_object_mean_position(depth_img, mask, RT_camera, cam_k)
    print(f"Object mean position (base_link): {mean_position}")
    
    is_far = np.linalg.norm(mean_position) > distance_threshold
    if is_far:
        print(f"Object is more than {distance_threshold} meters away.")
        point = find_point_at_distance(mean_position, target_distance=distance_threshold)
        return mean_position, is_far, point
    else:
        print("Object is within 2 meters.")
        return mean_position, is_far, None
    

def transform_mean_position(original_mean_position, RT_base_initial, RT_base_final):
    """Transform original mean position to new base_link frame using RT_base poses."""
    # Convert original mean position to homogeneous coordinates
    mean_hom = np.append(original_mean_position, 1)
    
    mean_new_baselink = np.linalg.inv(RT_base_final) @ RT_base_initial @ mean_hom
    
    return mean_new_baselink[:3]

def gaussian(x, mu, sig):
        return (
            1.0
            / (np.sqrt(2.0 * np.pi) * sig)
            * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )

def get_gaussian_interpolation_coefficients(num_points, scale=5):
    # method taken from https://github.com/robot-learning-freiburg/DITTO/blob/master/DITTO/mixing.py#L9
    # this defines a mixing function based on a gauss curve.
    # scales defines the steepness of the curve
    lt = num_points

    gs = gaussian(np.arange(lt), 0, lt / scale)
    mix = gs / (gs + gs[::-1])
    return mix, mix[::-1]

if __name__ == "__main__":
    complex = True
    if not complex:
        trajectory1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        trajectory2 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        trajectory2[:,2] += 1
    else:
    # generate more complex trajectories like curves
        t = np.linspace(0, 2*np.pi, 100)
    
        # Generate a spiral trajectory
        trajectory1 = np.zeros((100, 3))
        trajectory1[:,0] = t * np.cos(t) 
        trajectory1[:,1] = t * np.sin(t)
        trajectory1[:,2] = t/2

        # Generate a figure-8 trajectory
        trajectory2 = np.zeros((100, 3))
        trajectory2[:,0] = np.sin(t)
        trajectory2[:,1] = np.sin(t) * np.cos(t)
        trajectory2[:,2] = np.cos(t)

        trajectory2[:,2] += 1

    mix, mix_rev = get_gaussian_interpolation_coefficients(len(trajectory1), scale=4)
    print(mix)
    print(mix_rev)
    mix_combined = mix + mix_rev
    print(mix_combined)
    
    
    trajectory_combined = trajectory1 * mix[..., None] + trajectory2 * mix_rev[..., None]
    print(trajectory2)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], 'r')
    ax.plot3D(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], 'b')
    ax.plot3D(trajectory_combined[:, 0], trajectory_combined[:, 1], trajectory_combined[:, 2], 'g')
    ax.legend(['trajectory1', 'trajectory2', 'trajectory_combined'])
    plt.show()
