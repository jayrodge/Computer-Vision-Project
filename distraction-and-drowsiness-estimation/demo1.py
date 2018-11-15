import imutils
import cv2
import sys
import ddestimator

class demo1:

	FRAME_WIDTH = 450
	WINDOW_TITLE = "Distraction & Drowsiness Estimation using Kazemi & Sullivan's Ensemble of Regression Trees"

	PROCESS_INTERVAL = 50

	K_ESC = 27
	K_QUIT = ord('q')
	K_POINTS = ord('p')
	K_BOUNDING = ord('b')
	K_GAZE = ord('g')
	K_EYES = ord('e')
	K_MOUTH = ord('y')
	K_DD = ord('d')
	K_NONE = ord('n')
	K_REFRESH = ord('r')

	def __init__(self):
		cv2.namedWindow(demo1.WINDOW_TITLE)
		self.show_points = False
		self.show_bounding = False
		self.show_gaze = False
		self.show_ear = False
		self.show_yawn = False
		self.show_dd = True
		self.ddestimator = ddestimator.ddestimator()

	def run(self):
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			print("Unable to connect to camera.")
			return
		while self.cap.isOpened():
			self.key_strokes_handler()
			ret, frame = self.cap.read()
			if ret:
				frame = imutils.resize(frame, width=demo1.FRAME_WIDTH)
				frame = self.process_frame(frame)
				cv2.imshow(demo1.WINDOW_TITLE, frame)
				cv2.moveWindow(demo1.WINDOW_TITLE, 0, 0)

	def process_frame(self, frame=None):
		faces_loc = self.ddestimator.detect_faces(frame, None, True)
		if len(faces_loc) > 0:
			face_loc = faces_loc[0]
			points = self.ddestimator.pred_points_on_face(frame, face_loc)
			if self.show_points:
				frame = self.ddestimator.draw_points_on_face(frame, points, (0, 0, 255))
			euler, rotation, translation = self.ddestimator.est_head_dir(points)
			bc_2d_coords = self.ddestimator.proj_head_bounding_cube_coords(rotation, translation)
			if self.show_bounding:
				frame = self.ddestimator.draw_bounding_cube(frame, bc_2d_coords, (0, 0, 255))

		return frame

	def key_strokes_handler(self):
		pressed_key = cv2.waitKey(1) & 0xFF

		if pressed_key == demo1.K_ESC or pressed_key == demo1.K_QUIT:
			print('-> QUIT')
			self.cap.release()
			cv2.destroyAllWindows()
			sys.exit(0)

		elif pressed_key == demo1.K_POINTS:
			print('-> SHOW FACIAL LANDMARKS')
			self.show_points = True
			return None

		elif pressed_key == demo1.K_BOUNDING:
			print('-> SHOW BOUNDING CUBE FOR HEAD DIRECTION ESTIMATION')
			self.show_bounding = True
			return None

		elif pressed_key == demo1.K_GAZE:
			print('-> SHOW LINES FOR GAZE DIRECTION ESTIMATION')
			self.show_gaze = True
			return None

		elif pressed_key == demo1.K_EYES:
			print('-> SHOW EYE OPENNESS ESTIMATION')
			self.show_ear = True
			return None

		elif pressed_key == demo1.K_MOUTH:
			print('-> SHOW MOUTH OPENNESS ESTIMATION')
			self.show_yawn = True
			return None

		elif pressed_key == demo1.K_DD:
			self.show_dd = True
			return None

		elif pressed_key == demo1.K_NONE:
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_yawn = False
			self.show_dd = False
			return None

		elif pressed_key == demo1.K_REFRESH:
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_yawn = False
			self.show_dd = True
			return None
		else:
			return None

if __name__ == '__main__':
	demo1 = demo1()
	demo1.run()
