import cv2
import numpy as np
from pynput.keyboard import Controller

# --- Configuration ---
# Set up the keyboard controller
keyboard = Controller()

# The key to press when movement is in the left zone
KEY_LEFT = ','
# The key to press when movement is in the right zone
KEY_RIGHT = '.'
# Threshold
MIN_MOVEMENT_AREA = 1500

def process_frame(frame, fgbg, width):

    # 1. Apply background subtraction
    fgmask = fgbg.apply(frame)

    # 2. Apply erosion and dilation to remove noise and fill holes
    
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # 3. Find contours (i.e., detected moving objects)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variables to track the largest movement and its center
    max_area = 0
    movement_center_x = -1
    
    # Coordinates for the largest bounding box
    bbox = None 
    
    # 4. Loop through contours to find the largest movement above threshold
    for contour in contours:
        area = cv2.contourArea(contour)

        # Check if the area is above the minimum threshold AND is the largest so far
        if area > MIN_MOVEMENT_AREA and area > max_area:
            max_area = area
            # Calculate the bounding box for the largest movement
            (x, y, w, h) = cv2.boundingRect(contour)
            # Calculate the center x-coordinate of the movement
            movement_center_x = x + w // 2
            bbox = (x, y, w, h)

    # 5. Draw the bounding box for the single largest movement if one was found
    if bbox is not None:
        (x, y, w, h) = bbox
        # Draw a bounding box around ONLY the largest movement for visualization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    left_threshold_line = width // 3
    right_threshold_line = 2 * width // 3
    
    key_pressed = ""
    
    # Release both keys first to ensure a clean state before pressing
    keyboard.release(KEY_LEFT)
    keyboard.release(KEY_RIGHT)

    if movement_center_x != -1:
        if movement_center_x < left_threshold_line:
            # Movement is in the LEFT zone
            key_pressed = KEY_LEFT
            keyboard.press(KEY_LEFT)
            cv2.putText(frame, f"LEFT ZONE: '{KEY_LEFT}' pressed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif movement_center_x >= right_threshold_line:
            # Movement is in the RIGHT zone
            key_pressed = KEY_RIGHT
            keyboard.press(KEY_RIGHT)
            cv2.putText(frame, f"RIGHT ZONE: '{KEY_RIGHT}' pressed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Movement is in the MIDDLE/DEAD zone
            key_pressed = "DEAD ZONE"
            cv2.putText(frame, "DEAD ZONE: No key pressed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # No significant movement detected
        cv2.putText(frame, "No significant movement", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # The Left Threshold line (Red)
    cv2.line(frame, (left_threshold_line, 0), (left_threshold_line, frame.shape[0]), (0, 0, 255), 2)
    # The Right Threshold line (Red)
    cv2.line(frame, (right_threshold_line, 0), (right_threshold_line, frame.shape[0]), (0, 0, 255), 2)
    
    # Labels of zones
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


    # Initialize the Background Subtractor (Mixture of Gaussians)
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

            # Display the resulting frame and the movement mask
            cv2.imshow('Tri-Zone Gesture Key Press Control', display_frame)
            cv2.imshow('Movement Mask (Debug)', mask)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Ensure keys are released on exit
        keyboard.release(KEY_LEFT)
        keyboard.release(KEY_RIGHT)
        print("--- Application Closed ---")

if __name__ == "__main__":
    main()