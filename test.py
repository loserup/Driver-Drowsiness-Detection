import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
else:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Can't receive frame (stream end?).")
    else:
        print("[INFO] Frame received:", frame.shape)
cap.release()
