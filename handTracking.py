import cv2
import numpy as np
import mediapipe as mp

#Setup hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

#Get camera
cap = cv2.VideoCapture(0)

#Set drawing space
ret, frame = cap.read()
screenHeight, screenWidth, _ = frame.shape

#Drawing state
drawing = False
lastX, lastY = -1, -1

def isDrawing():
    #Get coordinates for relavent points
    DIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    DIPx, DIPy, DIPz = DIP.x, DIP.y, DIP.z

    PIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    PIPx, PIPy, PIPz = PIP.x, PIP.y, PIP.z

    MCP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    MCPx, MCPy, MCPz = MCP.x, MCP.y, MCP.z   

    TIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = round(TIP.x * screenWidth), round(TIP.y * screenHeight)


    #v1 is the vector from the index finger DIP to PIP
    v1 = np.array([DIPx - PIPx, DIPy - PIPy, DIPz - PIPz])

    #v2 is the vector from the index finger PIP to MCP
    v2 = np.array([PIPx - MCPx, PIPy - MCPy, PIPz - MCPz])

    #Calculate difference in angle
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (mag_v1 * mag_v2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    #Return state, true if straight
    if angle_degrees < 30:
        return True, x, y
    return False, x, y


def draw() -> None:
    global drawing, lastX, lastY
    event, x, y = isDrawing()

    #Reflect
    x = screenWidth - x

    #Check States
    if event and not drawing:
        drawing = True
        lastX, lastY = x, y
    elif event and drawing:
        if lastX != -1 and lastY != -1:            
            cv2.line(canvas, (lastX, lastY), (x, y), (0, 255, 0), 2)
        lastX, lastY = x, y
    elif not event:
        drawing = False
        lastX, lastY = -1, -1

def clear():
    #Get middle finger tip and thumb tip
    pTIP = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    tTIP = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    #Set coordinates
    px, py, pz = pTIP.x, pTIP.y, pTIP.z
    tx, ty, tz = tTIP.x, tTIP.y, tTIP.z

    #Magnitudes -mM, mT
    mP = np.sqrt(px**2 + py**2 + pz**2)
    mT = np.sqrt(tx**2 + ty**2 + tz**2)

    diff = mP - mT
    if diff < 0.04 and diff > -0.04:
        canvas[:] = 0


canvas = np.zeros((screenHeight, screenWidth, 3), np.uint8)
cv2.namedWindow("Screen Draw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Screen Draw", screenWidth, screenHeight)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw()
            clear()
            for id, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * screenWidth), int(landmark.y * screenHeight) #Get x, y for each landmark

                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        

    cv2.imshow("Hand Tracking", frame)
    cv2.imshow("Screen Draw", canvas)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()