import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Indices for eyes (from Mediapipe's 468 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye_points, landmarks):
    p1 = np.array(landmarks[eye_points[1]])
    p2 = np.array(landmarks[eye_points[5]])
    p3 = np.array(landmarks[eye_points[2]])
    p4 = np.array(landmarks[eye_points[4]])
    p5 = np.array(landmarks[eye_points[0]])
    p6 = np.array(landmarks[eye_points[3]])

    A = np.linalg.norm(p1 - p2)
    B = np.linalg.norm(p3 - p4)
    C = np.linalg.norm(p5 - p6)
    ear = (A + B) / (2.0 * C)
    return ear

# Draw eye landmarks on the frame
def draw_eye(frame, eye_points, landmarks, color=(0, 255, 255), thickness=1):
    points = [landmarks[i] for i in eye_points]

    # Draw circles on each landmark
    for point in points:
        cv2.circle(frame, point, 2, color, -1)

    # Connect points to form eye outline (using lines between consecutive points + closing the loop)
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]
        cv2.line(frame, start_point, end_point, color, thickness)

# EAR threshold
EAR_THRESHOLD = 0.25

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            status = "Eyes Open" if avg_ear > EAR_THRESHOLD else "Eyes Closed"
            color = (0, 255, 0) if avg_ear > EAR_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Draw eyes
            draw_eye(frame, LEFT_EYE, landmarks)
            draw_eye(frame, RIGHT_EYE, landmarks)

    cv2.imshow("Eye State Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
