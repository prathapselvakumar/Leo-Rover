import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HandEyeCalibration:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.H_cam2gripper = None
        
    def set_camera_intrinsics(self, camera_matrix, dist_coeffs):
        """
        Set camera intrinsic parameters
        
        Args:
            camera_matrix (np.ndarray): 3x3 camera matrix
            dist_coeffs (np.ndarray): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
    def add_aruco_board_detection(self, image):
        """
        Detect ArUco board in the image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (corners, ids, rejected) from ArUco detection
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard_create(
            squaresX=5,
            squaresY=7,
            squareLength=0.04,
            markerLength=0.02,
            dictionary=aruco_dict)
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if len(corners) > 0:
            # Refine detected markers
            corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                gray, corners, ids, rejected, self.camera_matrix, self.dist_coeffs)
                
            # Get charuco corners
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
                
            return charuco_corners, charuco_ids, board
            
        return None, None, None
    
    def estimate_pose(self, corners, ids, board):
        """
        Estimate pose of the ArUco board
        
        Args:
            corners: Corners from ArUco detection
            ids: IDs from ArUco detection
            board: ArUco board object
            
        Returns:
            tuple: (rvec, tvec) rotation vector and translation vector
        """
        if ids is None or len(ids) < 5:  # Minimum 5 points for pose estimation
            return None, None
            
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, board, self.camera_matrix, 
            self.dist_coeffs, None, None)
            
        if ret:
            return rvec, tvec
        return None, None
    
    def collect_calibration_data(self, gripper_poses, images):
        """
        Collect data for hand-eye calibration
        
        Args:
            gripper_poses: List of gripper poses (4x4 transformation matrices)
            images: List of images for each gripper pose
            
        Returns:
            tuple: (rvecs_gripper2base, tvecs_gripper2base, 
                   rvecs_target2cam, tvecs_target2cam)
        """
        rvecs_gripper2base = []
        tvecs_gripper2base = []
        rvecs_target2cam = []
        tvecs_target2cam = []
        
        for img, gripper_pose in zip(images, gripper_poses):
            # Convert gripper pose to rotation vector and translation
            r_gripper2base = R.from_matrix(gripper_pose[:3, :3])
            t_gripper2base = gripper_pose[:3, 3]
            rvec_gripper2base = r_gripper2base.as_rotvec()
            
            # Detect ArUco board and estimate pose
            corners, ids, board = self.add_aruco_board_detection(img)
            if corners is None:
                continue
                
            rvec_target2cam, tvec_target2cam = self.estimate_pose(corners, ids, board)
            if rvec_target2cam is None:
                continue
                
            # Store the data
            rvecs_gripper2base.append(rvec_gripper2base)
            tvecs_gripper2base.append(t_gripper2base)
            rvecs_target2cam.append(rvec_target2cam.reshape(-1))
            tvecs_target2cam.append(tvec_target2cam.reshape(-1))
            
        return (np.array(rvecs_gripper2base), np.array(tvecs_gripper2base),
                np.array(rvecs_target2cam), np.array(tvecs_target2cam))
    
    def calibrate_hand_eye(self, rvecs_gripper2base, tvecs_gripper2base,
                          rvecs_target2cam, tvecs_target2cam):
        """
        Perform hand-eye calibration
        
        Args:
            rvecs_gripper2base: List of rotation vectors (gripper to base)
            tvecs_gripper2base: List of translation vectors (gripper to base)
            rvecs_target2cam: List of rotation vectors (target to camera)
            tvecs_target2cam: List of translation vectors (target to camera)
            
        Returns:
            np.ndarray: 4x4 transformation matrix from camera to gripper
        """
        # Convert rotation vectors to rotation matrices
        R_gripper2base = [R.from_rotvec(rvec).as_matrix() 
                         for rvec in rvecs_gripper2base]
        t_gripper2base = [tvec.reshape(3, 1) for tvec in tvecs_gripper2base]
        
        R_target2cam = [R.from_rotvec(rvec).as_matrix() 
                       for rvec in rvecs_target2cam]
        t_target2cam = [tvec.reshape(3, 1) for tvec in tvecs_target2cam]
        
        # Perform hand-eye calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_PARK)
            
        # Create 4x4 transformation matrix
        self.H_cam2gripper = np.eye(4)
        self.H_cam2gripper[:3, :3] = R_cam2gripper
        self.H_cam2gripper[:3, 3] = t_cam2gripper.reshape(-1)
        
        return self.H_cam2gripper
    
    def transform_point(self, point_in_camera):
        """
        Transform a point from camera coordinates to gripper coordinates
        
        Args:
            point_in_camera (np.ndarray): 3D point in camera coordinates
            
        Returns:
            np.ndarray: 3D point in gripper coordinates
        """
        if self.H_cam2gripper is None:
            raise ValueError("Hand-eye calibration not performed yet")
            
        # Convert to homogeneous coordinates
        point_hom = np.ones(4)
        point_hom[:3] = point_in_camera
        
        # Transform to gripper coordinates
        point_gripper = self.H_cam2gripper @ point_hom
        return point_gripper[:3]
    
    def visualize_calibration(self, rvecs_gripper2base, tvecs_gripper2base,
                            rvecs_target2cam, tvecs_target2cam):
        """
        Visualize the calibration results
        
        Args:
            rvecs_gripper2base: List of rotation vectors (gripper to base)
            tvecs_gripper2base: List of translation vectors (gripper to base)
            rvecs_target2cam: List of rotation vectors (target to camera)
            tvecs_target2cam: List of translation vectors (target to camera)
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot gripper poses
        for rvec, tvec in zip(rvecs_gripper2base, tvecs_gripper2base):
            R_mat = R.from_rotvec(rvec).as_matrix()
            self._draw_frame(ax, R_mat, tvec, 'g', 'Gripper')
        
        # Plot target poses in camera frame
        for rvec, tvec in zip(rvecs_target2cam, tvecs_target2cam):
            R_mat = R.from_rotvec(rvec).as_matrix()
            self._draw_frame(ax, R_mat, tvec, 'b', 'Target')
        
        # Plot camera to gripper transformation
        if self.H_cam2gripper is not None:
            self._draw_frame(ax, self.H_cam2gripper[:3, :3], 
                           self.H_cam2gripper[:3, 3], 'r', 'Camera')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Hand-Eye Calibration Visualization')
        plt.show()
    
    def _draw_frame(self, ax, R, t, color, label=None):
        """Helper function to draw a coordinate frame"""
        # Origin
        ax.scatter(t[0], t[1], t[2], color=color, label=label)
        
        # Axes
        axis_length = 0.1
        x_axis = t + R[:, 0] * axis_length
        y_axis = t + R[:, 1] * axis_length
        z_axis = t + R[:, 2] * axis_length
        
        ax.plot([t[0], x_axis[0]], [t[1], x_axis[1]], [t[2], x_axis[2]], 'r')
        ax.plot([t[0], y_axis[0]], [t[1], y_axis[1]], [t[2], y_axis[2]], 'g')
        ax.plot([t[0], z_axis[0]], [t[1], z_axis[1]], [t[2], z_axis[2]], 'b')


def main():
    # Example usage
    print("Hand-Eye Calibration Module")
    
    # Initialize calibration
    calib = HandEyeCalibration()
    
    # Example camera intrinsics (replace with actual values from camera calibration)
    camera_matrix = np.array([
        [600, 0, 320],
        [0, 600, 240],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(5)  # Replace with actual distortion coefficients
    
    calib.set_camera_intrinsics(camera_matrix, dist_coeffs)
    
    # In a real scenario, you would:
    # 1. Move the robot to different poses
    # 2. For each pose:
    #    - Get the gripper pose (from robot forward kinematics)
    #    - Capture an image of the calibration target
    # 3. Process all poses to perform calibration
    
    print("Hand-eye calibration module ready. Implement data collection and call calibration methods.")

if __name__ == "__main__":
    main()
