import cv2
import numpy as np

#Set Screen
screenWidth = 800
screenHeight = 600

canvas = np.zeros((screenHeight, screenWidth, 3), np.uint8)

#Drawing state
drawing = False
lastX, lastY = -1, -1


def draw(event, x, y, flags, param) -> None:
    global drawing, lastX, lastY
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        lastX, lastY = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if lastX != -1 and lastY != -1:            
            cv2.line(canvas, (lastX, lastY), (x, y), (0, 255, 0), 2)
        lastX, lastY = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        lastX, lastY = -1, -1


cv2.namedWindow("Screen Draw", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Screen Draw", 600,800)
cv2.setMouseCallback("Screen Draw", draw)

while True:
    cv2.imshow("Screen Draw", canvas)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
cv2.destroyAllWindows()



