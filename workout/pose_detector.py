import cv2
import mediapipe as mp
import math
from collections import deque

# =========================
# POSE DETECTOR
# =========================
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


# =========================
# ARM PROCESSING FUNCTION
# =========================
def process_arm(shoulder, elbow, wrist,
                angle_buffer, state,
                rep_count, shoulder_ref):

    angle = calculate_angle(shoulder, elbow, wrist)
    angle_buffer.append(angle)
    smooth_angle = sum(angle_buffer) / len(angle_buffer)

    posture = "OK"

    # READY
    if smooth_angle > 160:
        state = "DOWN"
        shoulder_ref = shoulder[1]

    # CURL
    if smooth_angle < 70 and state == "DOWN":
        rep_count += 1
        state = "UP"

        if shoulder_ref is not None:
            move = abs(shoulder[1] - shoulder_ref)
            if move > 40:
                posture = "SHOULDER MOVED"
            else:
                posture = "GOOD FORM"

    return smooth_angle, state, rep_count, shoulder_ref, posture


# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    # LEFT ARM STATE
    left_reps = 0
    left_state = "DOWN"
    left_buffer = deque(maxlen=5)
    left_shoulder_ref = None
    left_posture = "OK"

    # RIGHT ARM STATE
    right_reps = 0
    right_state = "DOWN"
    right_buffer = deque(maxlen=5)
    right_shoulder_ref = None
    right_posture = "OK"

    active_arm = "NONE"

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, results = detector.detect(frame)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # LEFT ARM POINTS
            l_shoulder = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
            l_elbow    = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_ELBOW)
            l_wrist    = get_point(frame, lm, mp_pose.PoseLandmark.LEFT_WRIST)

            # RIGHT ARM POINTS
            r_shoulder = get_point(frame, lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
            r_elbow    = get_point(frame, lm, mp_pose.PoseLandmark.RIGHT_ELBOW)
            r_wrist    = get_point(frame, lm, mp_pose.PoseLandmark.RIGHT_WRIST)

            # PROCESS LEFT ARM
            l_angle, left_state, left_reps, left_shoulder_ref, left_posture = \
                process_arm(l_shoulder, l_elbow, l_wrist,
                            left_buffer, left_state,
                            left_reps, left_shoulder_ref)

            # PROCESS RIGHT ARM
            r_angle, right_state, right_reps, right_shoulder_ref, right_posture = \
                process_arm(r_shoulder, r_elbow, r_wrist,
                            right_buffer, right_state,
                            right_reps, right_shoulder_ref)

            # DETECT ACTIVE ARM
            if l_angle < r_angle:
                active_arm = "LEFT"
            elif r_angle < l_angle:
                active_arm = "RIGHT"
            else:
                active_arm = "NONE"

            # DISPLAY ANGLES
            cv2.putText(frame, f"L Angle: {int(l_angle)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"R Angle: {int(r_angle)}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # DISPLAY COUNTS
        cv2.putText(frame, f"Left Reps: {left_reps}", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        cv2.putText(frame, f"Right Reps: {right_reps}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # DISPLAY STATUS
        cv2.putText(frame, f"Active Arm: {active_arm}", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
        cv2.putText(frame, f"L Posture: {left_posture}", (30, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,255), 2)
        cv2.putText(frame, f"R Posture: {right_posture}", (30, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,255), 2)

        cv2.imshow("AI-GymMate | Day 5 – Both Arms", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
