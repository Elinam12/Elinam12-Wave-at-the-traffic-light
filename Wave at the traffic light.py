import cv2
import numpy as np
import time

# Number of waves
WAVE_ NUMBER = 30  

# Distance to consider a movement as a wave
WAVE_DETECTION_RANGE = 50  

# Initialize traffic light state
traffic_light_color = "red"


def is_this_a_wave(movement):
    """
    Check if the movement pattern indicates a wave.
    """
    return len(movement) >= WAVE_NUMBER


def display_traffic_light(color):
    """
    Display the traffic light with the given color.
    """
    light = np.zeros((300, 100, 3), dtype="uint8")
    color_dict = {"red": (0, 0, 255), "green": (0, 255, 0)}
    if color in color_dict:
        cv2.circle(light, (50, 150 if color == "green" else 50), 30, color_dict[color], -1)
    cv2.imshow("Traffic Light", light)


def trigger_traffic_light():
    """
    Trigger the traffic light to turn green and then back to red.
    """
    traffic_light_color = "green"
    show_traffic_light(traffic_light_color)
    print("Traffic light turned green!")
    time.sleep(5)  # Simulate the green light duration
    traffic_light_color = "red"
    display_traffic_light(traffic_light_color)
    print("Traffic light turned red!")


def detect_handwave():
    """
    Detect handwave using the webcam and trigger the traffic light.
    """
    cap = cv2.VideoCapture(0)
    movement = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            movement.append((cx, cy))
            if len(movement) > WAVE_THRESHOLD:
                movement.pop(0)
                if is_wave(movement):
                    print("Handwave detected! Triggering pedestrian crossing signal...")
                    trigger_traffic_light()
                    movement = []

        cv2.imshow("Frame", frame)
        cv2.imshow("Thresh", thresh)
        display_traffic_light(traffic_light_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_handwave()


