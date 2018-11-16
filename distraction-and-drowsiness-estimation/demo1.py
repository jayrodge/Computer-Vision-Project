import imutils
import cv2
import sys
import time
import ddestimator
import pandas as pd

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
	K_SAVE_LOG = ord('l')
	K_HELP = ord('h')

	LOG_PATH = './kss_%ts.csv'

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
			eye_closeness = self.ddestimator.get_eye_closedness_over_time(points)
			yawn = self.ddestimator.get_mouth_openess_over_time(points)
			# TODO: do calibration only once
			has_calibration, _, meds = self.ddestimator.get_med_eulers()
			#if has_calibration:
			#	print(str(meds*-1))

			if self.show_bounding:
				bc_2d_coords = self.ddestimator.proj_head_bounding_cube_coords(rotation, translation)
				frame = self.ddestimator.draw_bounding_cube(frame, bc_2d_coords, (0, 0, 255), euler)
			_, _, gaze_D = self.ddestimator.est_gaze_dir(points)

			if self.show_gaze:
				gl_2d_coords = self.ddestimator.proj_gaze_line_coords(rotation, translation, gaze_D)
				self.ddestimator.draw_gaze_line(frame, gl_2d_coords, (0, 255, 0), gaze_D)
			if self.show_ear:
				frame=self.ddestimator.draw_eye_lines(frame,points[42:48],points[36:42])
			if self.show_yawn:
				frame = self.ddestimator.draw_mouth(frame,points[60:68])
			if self.show_dd:
				head_distraction, _, _ = self.ddestimator.est_head_dir_over_time()
				if not head_distraction:
					gaze_distraction, _, _ = self.ddestimator.est_gaze_dir_over_time()
				else:
					gaze_distraction = False
				if head_distraction:
					cv2.putText(frame, "DISTRACTED", (20,20),
								cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=1)
				if gaze_distraction:
					cv2.putText(frame, "distracted", (20,20),
								cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=1)

				if yawn:
					cv2.putText(frame,"DROWSY",(350,20),
					            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=1)	
				if eye_closeness:
					cv2.putText(frame,"drowsy",(375,20),
					            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=1,bottomLeftOrigin=False)
				kss = self.ddestimator.calc_kss()
				if kss is not None:
					kss_int = int(round(kss*10))
					frame = self.ddestimator.draw_progress_bar(frame, 140, 35, kss, str(kss_int))

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
			print('-> SHOW DROWSINESS & DISTRACTION ESTIMATIONS')
			self.show_dd = True
			return None

		elif pressed_key == demo1.K_NONE:
			print('-> SHOW NO ESTIMATIONS')
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_yawn = False
			self.show_dd = False
			return None

		elif pressed_key == demo1.K_REFRESH:
			print('-> RESET SHOW TO DEFAULT')
			self.show_bounding = False
			self.show_gaze = False
			self.show_ear = False
			self.show_yawn = False
			self.show_dd = True
			return None

		elif pressed_key == demo1.K_SAVE_LOG:
			print('-> SAVE LOG FILE WITH KSS ESTIMATIONS')
			kss_log = self.ddestimator.fetch_log('kss')
			ts = int(round(time.time() * 1000))
			path = (demo1.LOG_PATH).replace('%ts', str(ts))
			print("\t"+path)
			kss_log.to_csv(path)
			return None

		# TODO: help screen
		elif pressed_key == demo1.K_HELP:
			return None

		else:
			return None

if __name__ == '__main__':
	demo1 = demo1()
	demo1.run()
