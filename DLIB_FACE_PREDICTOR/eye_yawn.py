from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

#For dlib’s 68-point facial landmark detector:
# FACIAL_LANDMARKS_68_IDXS = OrderedDict([
# 	("mouth", (48, 68)),
# 	("right_eyebrow", (17, 22)),
# 	("left_eyebrow", (22, 27)),
# 	("right_eye", (36, 42)),
# 	("left_eye", (42, 48)),
# 	("nose", (27, 36)),
# 	("jaw", (0, 17))
# ])

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25 # For EAR
frame_check = 40
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

#Left Eye Co-ordinates
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] 
#Right Eye Co-ordinates
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] 
#Mouth Co-ordinates
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"] 

cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

    #Looping through Multiple Faces
	for subject in subjects: 
        #Predicting the facial Landmarks
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)

        # Extracting the Left and Right Eye, Mouth
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

        #Calculating EYE ASPECT RATIO
		ear = (leftEAR + rightEAR) / 2.0

        # Calculating Distance 63rd and 67th facial landmark
		mouth_open=distance.euclidean(mouth[9], mouth[13])
		# print(mouth_open)
		
        # Highlighting Eyes and mouth for reference
		mouthHull = cv2.convexHull(mouth)
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) 
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouthHull],-1, (0, 255, 0), 1)

		#Applying Threshold
		if ear < thresh or mouth_open > 40 :
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************WAKE UP****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			flag = 0

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.stop()
