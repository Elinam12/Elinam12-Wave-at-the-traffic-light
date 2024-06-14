import cv2
import numpy as np
import time

def capture_video():
    cap = cv2.VideoCapture(0)  # 0 for the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Constants
WAVE_THRESHOLD = 30  # Number of frames for a wave
WAVE_DETECTION_RADIUS = 50  # Distance to consider a movement as a wave


def is_wave(movement):
    # Check if the movement pattern indicates a wave
    if len(movement) < WAVE_THRESHOLD:
        return False
    return True

def detect_handwave():
    cap = cv2.VideoCapture(0)
    movement = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # Threshold to get the binary image
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour which should be the hand
            cnt = max(contours, key=cv2.contourArea)
            # Get the bounding box of the hand
            x, y, w, h = cv2.boundingRect(cnt)
            # Get the center of the hand
            cx, cy = x + w // 2, y + h // 2

            # Record the movement
            movement.append((cx, cy))

            if len(movement) > WAVE_THRESHOLD:
                movement.pop(0)

                # Check if the recorded movement indicates a wave
                if is_wave(movement):
                    print("Handwave detected! Triggering pedestrian crossing signal...")
                    movement = []
                    time.sleep(5)  # Simulate the signal duration

        # Display the result
        cv2.imshow("Frame", frame)
        cv2.imshow("Thresh", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_handwave()



