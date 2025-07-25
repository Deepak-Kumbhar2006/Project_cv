import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 263]

def draw_eyes(image, landmarks, image_width, image_height):
    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
        landmark = landmarks.landmark[idx]
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)

                h, w, _ = frame.shape
                draw_eyes(frame, face_landmarks, w, h)

        cv2.imshow('Face Mesh with Eye Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
