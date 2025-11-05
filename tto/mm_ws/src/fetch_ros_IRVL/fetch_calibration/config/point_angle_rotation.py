import numpy as np

def rotation_matrix_roll(angle):
    """
    Create a rotation matrix for a given roll angle around the x-axis.
    
    :param angle: Angle in radians
    :return: 3x3 rotation matrix
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

# Define the roll angle in radians
roll_angle = -0.8287

# Create the rotation matrix
R = rotation_matrix_roll(roll_angle)

# Original point (example point; you can change this)
point = np.array([0.1044+0.0333, -0.0355, 0.0396])  # Example: a point at (1, 0, 0)

# Apply rotation to the point
rotated_point = np.dot(R, point)

print("Original Point:", point)
print("Rotated Point:", rotated_point)

# If you want to round the results for readability
print("Rotated Point (rounded):", np.round(rotated_point, 4))