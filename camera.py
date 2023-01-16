#camera.py
import cv2
import dlib
from imutils import face_utils
import imutils

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')
        success, image = self.video.read()
        image = imutils.resize(image, width=150)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for face in rects:
            x1,y1 = face.left(), face.top()
            x2,y2 = face.right(), face.bottom()
            imgoriginal = cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,0), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()