import numpy as np
np.set_printoptions(precision=6, suppress=True, floatmode='fixed')

# Config 0: Standard Fetch gripper
# No ati sensor; has old fingers
RT_wrist_to_fingertip_config0 = np.array([
    [1.000000,  0.000000,  0.000000,  0.196000],
    [0.000000,  1.000000,  0.000000,  0.000000],
    [0.000000,  0.000000,  1.000000,  0.000000],
    [0.000000,  0.000000,  0.000000,  1.000000]
])

# Config 1: Modified Fetch gripper
# ati sensor; has old fingers
RT_wrist_to_fingertip_config1 = np.array([
    [1.000000,  0.000000,  0.000000,  0.229750],
    [0.000000,  0.675835,  0.737053,  0.000000],
    [0.000000, -0.737053,  0.675835,  0.000000],
    [0.000000,  0.000000,  0.000000,  1.000000]
])

# Config 2: Modified Fetch gripper
# ati sensor; has umi fingers
# NOTE: The urdf tf for fingertop link is not accurate. This tf is from the obj
RT_wrist_to_fingertip_config2 = np.array([
    [1.000000,  0.000000,  0.000000,  0.311000],
    [0.000000,  0.675835,  0.737053,  0.000000],
    [0.000000, -0.737053,  0.675835,  0.000000],
    [0.000000,  0.000000,  0.000000,  1.000000]
])

RT_gripper_configs = {0: RT_wrist_to_fingertip_config0, 1: RT_wrist_to_fingertip_config1 , 2: RT_wrist_to_fingertip_config2}