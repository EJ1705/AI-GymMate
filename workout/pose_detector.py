import cv2
import mediapipe as mp
import math
from collections import deque

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame, results


mp_pose = mp.solutions.pose

def get_point(frame, lm, landmark):
    h, w, _ = frame.shape
    p = lm[landmark.value]
    return int(p.x * w), int(p.y * h)


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    return math.degrees(math.acos(dot / (mag_ba * mag_bc)))


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    rep_count = 0
    state = "DOWN"
    angle_buffer = deque(maxlen=5)

    shoulder_reference_y = None
    posture_status = "OK"

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, results = detector.detect(frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            shoulder = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
            elbow    = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_ELBOW)
            wrist    = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_WRIST)

            angle = calculate_angle(shoulder, elbow, wrist)
            angle_buffer.append(angle)
            smooth_angle = sum(angle_buffer) / len(angle_buffer)

            # READY POSITION
            if smooth_angle > 160:
                state = "DOWN"
                shoulder_reference_y = shoulder[1]
                posture_status = "OK"

            # CURL POSITION
            if smooth_angle < 70 and state == "DOWN":
                rep_count += 1
                state = "UP"

                # posture feedback (NOT blocking)
                if shoulder_reference_y is not None:
                    move = abs(shoulder[1] - shoulder_reference_y)
                    if move > 40:
                        posture_status = "SHOULDER MOVED"
                    else:
                        posture_status = "GOOD FORM"

            # DISPLAY TEXT
            cv2.putText(frame, f"Angle: {int(smooth_angle)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(frame, f"State: {state}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.putText(frame, f"Posture: {posture_status}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        cv2.putText(frame, f"Reps: {rep_count}", (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("AI-GymMate | Clear Debug Mode", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
