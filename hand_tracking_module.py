import cv2
import mediapipe as mp
import time


# Create a class called HandDetector(maintain camel casing)
class HandDetector:
    # Create initialization method for the class and pass the arguments of Hands() class
    def __init__(self, mode=False, max_hands=2, model_com=1, det_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_com = model_com
        self.det_con = det_con
        self.track_con = track_con

        self.lm_list = None
        self.results = None

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_com, self.det_con, self.track_con)

        self.tip_id = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for identity, landMark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landMark.x * w), int(landMark.y * h)
                self.lm_list.append([identity, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        fingers = []

        # Thumb
        if self.lm_list[self.tip_id[0]][1] < self.lm_list[self.tip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for identity in range(1, 5):
            if self.lm_list[self.tip_id[identity]][2] < self.lm_list[self.tip_id[identity]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    frame = cv2.VideoCapture(0)

    detector = HandDetector()

    previous_time = 0

    while True:
        (success, img) = frame.read()
        rgb_img = detector.find_hands(img)
        detector.find_position(rgb_img)
        # lm_list = detector.find_position(rgb_img)
        # if len(lm_list) != 0:
        #     print(lm_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, 'FPS: ', (1000, 68), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, str(int(fps)), (1130, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
