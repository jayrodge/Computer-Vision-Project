import numpy as np
import imutils
import dlib
import cv2
import time
import math
import pandas as pd

class ddestimator:

	#JAY RODGE ===============================================

	TRAINED_MODEL_PATH = './shape_predictor_68_face_landmarks.dat'

	def __init__(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(ddestimator.TRAINED_MODEL_PATH)
		self.start_time = int(round(time.time() * 1000))
		self.log = pd.DataFrame(data=[], columns=['ts','key','value'])
		self.log.set_index(['ts', 'key'])

	# Used the following code as reference: http://dlib.net/face_landmark_detection.py.html
	def detect_faces(self,  frame, resize_to_width=None, use_gray=True):
		# Faster prediction when frame is resized
		if resize_to_width is not None:
			frame = imutils.resize(frame, width=resize_to_width)
		# If use_gray = True then convert frame used for detection in to grayscale
		if use_gray:
			dframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			dframe = frame

		#Detect faces in frame
		faces_loc = self.detector(dframe, 0)

		return faces_loc

	def dlib_shape_to_points(self, shape, dtype=np.int32):
		points = np.zeros((68, 2), dtype=dtype)

		for j in range(0, 68):
			points[j] = (shape.part(j).x,shape.part(j).y)

		return points

	def pred_points_on_face(self, frame, face_loc, use_gray=True):
		# If use_gray = True then convert frame used for prediction in to grayscale
		if use_gray:
			pframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			pframe = frame

		shape = self.predictor(pframe, face_loc)
		points = self.dlib_shape_to_points(shape)
		return points

	def draw_points_on_face(self, frame, points, color):
		for (x, y) in points:
			cv2.circle(frame, (x, y), 1, color, -1)
		return frame

	# SERG MASIS ===============================================

	# These are the estimated 3D positions for 2D image points 17,21,22,26,36,39,42,45,31,35,48,54,57 & 8
	# taken from this model http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp (line 69)
	FACE_3D_ANCHOR_PTS = np.float32([[6.825897, 6.760612, 4.402142],
                                     [1.330353, 7.122144, 6.903745],
                                     [-1.330353, 7.122144, 6.903745],
                                     [-6.825897, 6.760612, 4.402142],
                                     [5.311432, 5.485328, 3.987654],
                                     [1.789930, 5.393625, 4.413414],
                                     [-1.789930, 5.393625, 4.413414],
                                     [-5.311432, 5.485328, 3.987654],
                                     [2.005628, 1.409845, 6.165652],
                                     [-2.005628, 1.409845, 6.165652],
                                     [2.774015, -2.080775, 5.048531],
                                     [-2.774015, -2.080775, 5.048531],
                                     [0.000000, -3.116408, 6.097667],
                                     [0.000000, -7.415691, 4.070434]])

	# Retrieved these matrices with OpenCV's camera calibration method
	# method https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	CAMERA_CALIBRATION_MATRIX = np.float32([[653.0839, 0., 319.5],
											[0., 653.0839, 239.5],
											[0., 0., 1.]])

	#k1, k2, p1, p2, k3
	CAMERA_DISTORTION_COEFFICIENTS = np.float32([[0.070834633684407095],
												 [0.0691402],
												 [0.],
												 [0.],
												 [-1.3073460323689292]])

	# 8 Point Bounding Cube Coordinates
	BOUNDING_CUBE_3D_COORDS = np.float32([[10.0, 10.0, 10.0],
										[10.0, 10.0, -10.0],
										[10.0, -10.0, -10.0],
										[10.0, -10.0, 10.0],
										[-10.0, 10.0, 10.0],
										[-10.0, 10.0, -10.0],
										[-10.0, -10.0, -10.0],
										[-10.0, -10.0, 10.0]])

	def euler_decomposition(self, projmat):
		sin_x = math.sqrt(projmat[0, 0] * projmat[0, 0] + projmat[1, 0] * projmat[1, 0])
		is_singular = sin_x < 0.000001

		if not is_singular:
			x = math.degrees(math.atan2(projmat[2, 1], projmat[2, 2]))
			y = math.degrees(math.atan2(-projmat[2, 0], sin_x))
			z = math.degrees(math.atan2(projmat[1, 0], projmat[0, 0]))
		else:
			x = math.degrees(math.atan2(-projmat[1, 2], projmat[1, 1]))
			y = math.degrees(math.atan2(-projmat[2, 0], sy))
			z = 0

		euler = np.array([x, y, z]).T
		return euler

	'''
	3D -> 2D point translation from 14/58 'anchor' points of 3D model to 14/68 points of 2D model using this chart (0 indexed)
	33 -> 17 (Left corner left eyebrow)
	29 -> 21 (Right corner left eyebrow)
	34 -> 22 (Left corner right eyebrow)
	38 -> 26 (Right corner right eyebrow)
	13 -> 36 (Left corner left eye)
	17 -> 39 (Right corner left eye)
	25 -> 42 (Left corner right eye)
	21 -> 45 (Left corner right eye)
	55 -> 31 (Left bottom nose)
	49 -> 35 (Right bottom nose)
	43 -> 48 (Left corner mouth)
	39 -> 54 (Right corner mouth)
	45 -> 57 (Bottom center of mouth)
	6 -> 8 (Center of Chin)
	'''
	def est_head_dir(self, points):
		face_2d_anchor_pts = np.array([points[17], points[21], points[22], points[26], points[36], points[39], points[42], points[45], points[31], points[35], points[48], points[54], points[57], points[8]], dtype=np.float32)

		# Get rotation and translation vectors for points and taking in account camera parameters
		_, rotvec, transvec = cv2.solvePnP(ddestimator.FACE_3D_ANCHOR_PTS,
											face_2d_anchor_pts,
											ddestimator.CAMERA_CALIBRATION_MATRIX,
											ddestimator.CAMERA_DISTORTION_COEFFICIENTS)

		# Get rotation matrix with rotation vector
		rotmat, _ = cv2.Rodrigues(rotvec)

		# Get projection matrix by concatenating rotation matrix and translation vector
		projmat = np.hstack((rotmat, transvec))

		# Get Euler angle from projection matrix
		euler = self.euler_decomposition(projmat)

		#print(str(self.push_to_log('euler_x', euler[0])))

		return euler, rotvec, transvec

	def proj_head_bounding_cube_coords(self, rotation, translation):
		# Project bounding box points using rotation and translation vectors and taking in account camera parameters
		bc_2d_coords, _ = cv2.projectPoints(ddestimator.BOUNDING_CUBE_3D_COORDS,
											rotation,
											translation,
											ddestimator.CAMERA_CALIBRATION_MATRIX,
											ddestimator.CAMERA_DISTORTION_COEFFICIENTS)
		return bc_2d_coords

	def draw_bounding_cube(self, frame, bc_2d_coords, color):
		bc_2d_coords = bc_2d_coords.reshape(8, 2)
		for from_pt, to_pt in np.array([[0, 1], [1, 2], [2, 3], [3, 0],
										[4, 5], [5, 6], [6, 7], [7, 4],
										[0, 4], [1, 5], [2, 6], [3, 7]]):
			cv2.line(frame, tuple(bc_2d_coords[from_pt]), tuple(bc_2d_coords[to_pt]), color)
		#                 cv2.putText(frame, "X: " + "{:7.2f}".format(euler[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
		#                             0.5, (0, 0, 0), thickness=1)
		#                 cv2.putText(frame, "Y: " + "{:7.2f}".format(euler[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
		#                             0.5, (0, 0, 0), thickness=1)
		#                 cv2.putText(frame, "Z: " + "{:7.2f}".format(euler[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
		#                             0.5, (0, 0, 0), thickness=1)
		return frame

	def est_gaze_dir(self, points):
		return None

	def draw_gaze_lines(self, lines):
		return None

	def proj_gaze_lines_coords(self, rotation, translation):
		return None

	def push_to_log(self, key, value):
		ts = int(round(time.time() * 1000))
		#self.log.loc[-1] = [ts, type, value]
		self.log = self.log.append({'ts': ts, 'key':key, 'value':value}, ignore_index=True)
		return self.log.count()

	def purge_from_log(self, ts_threshold, key):
		ts = int(round(time.time() * 1000)) - ts_threshold
		self.log = self.log.drop(df[(self.log.ts < ts) & (df.self.key = key)].index)
		return self.log.count()

	# JAY RODGE ===============================================
	def est_eye_closedness(self, points):
		return None

	def get_ear(self, eye_points):
		return None

	def get_eye_closedness_over_time(self, time=2000, threshold=None):
		return None

	def est_mouth_openess(self, points):
		return None

	def get_mouth_openess_over_time(self, time=4500, threshold=None):
		return None

