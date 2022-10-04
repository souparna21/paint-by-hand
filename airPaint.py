import cv2
import numpy as np
import os
import hand_tracking_module as htm

folder_path = "brushes"
my_list = os.listdir(folder_path)
overlay_list = []

for img_path in my_list:
    img = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(img)

top_img = overlay_list[4]
draw_color = (107, 250, 211)
brush_thickness = 15
eraser_thickness = 50
# xp, yp = 0, 0

frame = cv2.VideoCapture(0)
frame.set(3, 1280)
frame.set(4, 720)

detect_hand = htm.HandDetector(det_con=0.85)

# Create a canvas
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    (success, img) = frame.read()
    # Flip the image
    img = cv2.flip(img, 1)

    # Draw the landmarks on hand
    img = detect_hand.find_hands(img, draw=False)
    lm_list = detect_hand.find_position(img, draw=False)

    if len(lm_list) != 0:
        # Tip of index finger and middle fingers
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # Check which finger is up
        up_fingers = detect_hand.fingers_up()
        # print(up_fingers)

        # Selection mode - two finger are up
        if up_fingers[1] and up_fingers[2]:
            xp, yp = 0, 0
            # Checking for the click
            if y1 < 100:
                if 450 < x1 < 530:
                    top_img = overlay_list[2]
                    draw_color = (194, 222, 104)
                elif 550 < x1 < 650:
                    top_img = overlay_list[3]
                    draw_color = (135, 79, 224)
                elif 700 < x1 < 800:
                    top_img = overlay_list[0]
                    draw_color = (129, 203, 255)
                elif 900 < x1 < 1050:
                    top_img = overlay_list[1]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)

        # Drawing mode
        if up_fingers[1] and up_fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

            xp, yp = x1, y1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    # Convert it to binary image and also inverse it
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    img[0:100, 0:1280] = top_img

    # Add two images
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)

    # cv2.imshow("Canvas", img_canvas)
    cv2.imshow("My Canvas", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
