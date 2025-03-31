import cv2

print("OpenCV version:", cv2.__version__)
print("Has face module:", hasattr(cv2, "face"))
print("Has VideoCapture:", hasattr(cv2, "VideoCapture"))

# Try to use face recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("LBPHFaceRecognizer loaded successfully ✅")
except Exception as e:
    print("Error loading recognizer:", e)

# Try opening camera
cam = cv2.VideoCapture(0)
if cam.isOpened():
    print("Camera opened successfully ✅")
    cam.release()
else:
    print("Failed to open camera ❌")
