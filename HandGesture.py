import cv2
import time
import mediapipe as mp
from pynput.keyboard import Controller 


keyboard = Controller()
KEY_PRESS_COORD = {
    1: '1',  
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
}

# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils


# Finger Tip and Joint Indices
# TIP_IDS = [Thumb_Tip, Index_Tip, Middle_Tip, Ring_Tip, Pinky_Tip]
TIP_IDS = [4, 8, 12, 16, 20]

# Mapping of a finger's tip ID to its lower joint
# TIP_TO_JOINT: {Tip_ID: Joint_ID}
PIP_JOINT_IDS = [2, 6, 10, 14, 18]

def count_extended_fingers(hand_landmarks, is_right_hand):
    if not hand_landmarks:
        return 0

    finger_count = 0
    landmarks = hand_landmarks.landmark
    
    
    # Checking counts for four fingers
    # Finger is up: The tip is higher than the base joint
    for tip_index in TIP_IDS[1:]: 
        joint_index = tip_index - 2 

        # Y-axis is inverted: smaller Y means higher on screen
        if landmarks[tip_index].y < landmarks[joint_index].y:
            finger_count += 1
            
    # Checking counts for thumbs
    # Thumb is up: The tip joint (4) is horizontally further out than the PIP joint (2)
    thumb_tip_x = landmarks[TIP_IDS[0]].x   
    thumb_mcp_x = landmarks[TIP_IDS[0] - 2].x 
    
    # For a right hand 
    if is_right_hand:
        # A primary indicator of an open thumb, regardless of vertical tilt.
        if thumb_tip_x > thumb_mcp_x * 1.05:
             finger_count += 1
    
    # For a left hand 
    else: 
        # A primary indicator of an open thumb, tip extends left/smaller X.
        if thumb_tip_x < thumb_mcp_x * 0.95:
            finger_count += 1
            
    return finger_count


capture = cv2.VideoCapture(0)

# Initializing for calculating the FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))

    h, w, c = frame.shape
    
    # Calculate the Y-axis for threshold
    TOP_THIRD_THRESHOLD_AREA = 1.0 / 3.0

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize counts to zero
    right_count = 0
    left_count = 0

    # Checking if a hand is in the threshold area
    def is_in_top_third(landmarks):
        if not landmarks:
            return False
        # Use the wrist landmark as a reference point
        wrist_y_norm = landmarks.landmark[0].y
        return wrist_y_norm < TOP_THIRD_THRESHOLD_AREA
        
    
    # Process Right Hand
    if results.right_hand_landmarks and is_in_top_third(results.right_hand_landmarks):
        right_count = count_extended_fingers(results.right_hand_landmarks, is_right_hand=True)
    
    # Process Left Hand
    if results.left_hand_landmarks and is_in_top_third(results.left_hand_landmarks):
        left_count = count_extended_fingers(results.left_hand_landmarks, is_right_hand=False)
        
    total_count = right_count + left_count
    
    # Keyboard Control Input
    if total_count in KEY_PRESS_COORD:
        key = KEY_PRESS_COORD[total_count]
        keyboard.press(key)
        # Display the key pressed prominently
        cv2.putText(image, f"KEY PRESSED: {key}", (w - 300, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)


    # Label for 1/3 Threshold
    y_pixel_threshold = int(h * TOP_THIRD_THRESHOLD_AREA)
    cv2.line(image, (0, y_pixel_threshold), (w, y_pixel_threshold), (0, 255, 255), 2)
    cv2.putText(image, "Threshold Area", (w - 300, y_pixel_threshold - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Label for right hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Label for left hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    
    # Display FPS and Counts
    cv2.putText(image, str(int(fps))+" FPS", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, f"R Hand: {right_count}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.putText(image, f"L Hand: {left_count}", (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.putText(image, f"TOTAL: {total_count}", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Facial and Hand Landmarks", image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()