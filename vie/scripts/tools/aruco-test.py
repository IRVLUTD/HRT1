import pyrealsense2 as rs
import cv2
import numpy as np

# Replace with your camera intrinsics
camera_matrix = np.array([
    [527.8869068647631, 0.0, 321.7148665756361],
    [0.0, 524.7942507494529, 230.2819198622499],
    [0.0, 0.0, 1.0],
])
dist_coeffs = np.zeros(5)  # Assume no distortion (set to zero if unknown)

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream

# Start the RealSense pipeline
pipeline.start(config)

frame_count = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Skip frames to reduce the processing load
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        
        # Convert RealSense frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            # Estimate pose for all detected markers at once
            # Using estimatePoseSingleMarkers to handle all markers in one go
            print(cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs))
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs)

            # Draw axes for all detected markers
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Display the image
            cv2.imshow("RealSense ArUco Marker Detection", color_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
