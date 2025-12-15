import cv2
import numpy as np
from pynput.keyboard import Controller

keyboard = Controller()

# Key Maps
KEY_LEFT = ','
KEY_RIGHT = '.'

# Threshold
MIN_MOVEMENT_AREA = 1500

def process_frame(frame, fgbg, width):

    # Background subtraction
    fgmask = fgbg.apply(frame)

    # Erosion and dilation
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # 3. Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variables to track
    max_area = 0
    movement_center_x = -1
    
    # Largest bounding box coordinates
    bbox = None 
    
    # Loop through contours
    for contour in contours:
        area = cv2.contourArea(contour)

        # If the area is above the minimum threshold and is the largest so far
        if area > MIN_MOVEMENT_AREA and area > max_area:
            max_area = area
            # Calculate the bounding box for the largest movement
            (x, y, w, h) = cv2.boundingRect(contour)
            # Calculate the center x-coordinate of the movement
            movement_center_x = x + w // 2
            bbox = (x, y, w, h)

    # 5. Bounding box for single movement
    if bbox is not None:
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    left_threshold_line = width // 3
    right_threshold_line = 2 * width // 3
    
    key_to_press = None 
    status_text = "No significant movement"
    status_color = (255, 255, 255) 
    
    # If key pressed
    if movement_center_x != -1:
        if movement_center_x < left_threshold_line:
            # Left zone movement
            key_to_press = KEY_LEFT
            status_text = f"LEFT ZONE: '{KEY_LEFT}' pressed (Holding)"
            status_color = (0, 255, 0)
        elif movement_center_x >= right_threshold_line:
            # Right zone movement
            key_to_press = KEY_RIGHT
            status_text = f"RIGHT ZONE: '{KEY_RIGHT}' pressed (Holding)"
            status_color = (0, 255, 0)
        else:
            # Middle zone movement
            key_to_press = None
            status_text = "DEAD ZONE: No key pressed (Releasing)"
            status_color = (0, 255, 255)

    # Execute key presses
    if key_to_press == KEY_LEFT:
        keyboard.press(KEY_LEFT)
        keyboard.release(KEY_RIGHT)
    elif key_to_press == KEY_RIGHT:
        keyboard.press(KEY_RIGHT)
        keyboard.release(KEY_LEFT)
    else:
        keyboard.release(KEY_LEFT)
        keyboard.release(KEY_RIGHT)
        
    # Status display
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Left threshold line 
    cv2.line(frame, (left_threshold_line, 0), (left_threshold_line, frame.shape[0]), (0, 0, 255), 2)
    # Right threshold line
    cv2.line(frame, (right_threshold_line, 0), (right_threshold_line, frame.shape[0]), (0, 0, 255), 2)
    
    # Zone labels
    cv2.putText(frame, "LEFT", (left_threshold_line // 2 - 20, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "DEAD", (left_threshold_line + (width//6) - 20, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "RIGHT", (right_threshold_line + (width//6) - 20, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, fgmask

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Attempted to set resolution to 1280x720. Actual resolution: {actual_width}x{actual_height}")


    # Initialize the Background Subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print(f"--- Tri-Zone Gesture Control Initialized ---")
    print(f"Move body/hand to the LEFT (1st third) to press '{KEY_LEFT}'")
    print(f"Move body/hand to the RIGHT (3rd third) to press '{KEY_RIGHT}'")
    print("Keep movement in the MIDDLE (2nd third) to press nothing.")
    print("Press 'q' to exit the application.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            width = frame.shape[1]

            # Process the frame
            display_frame, mask = process_frame(frame, fgbg, width)

            # Resulting Frame
            cv2.imshow('Tri-Zone Gesture Key Press Control', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Key release on exit
        keyboard.release(KEY_LEFT)
        keyboard.release(KEY_RIGHT)
        print("--- Application Closed ---")

if __name__ == "__main__":
    main()