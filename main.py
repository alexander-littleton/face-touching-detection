from mediapipe.python.solutions import face_detection, drawing_utils, hands
import cv2

cap = cv2.VideoCapture(0)
i = 0
with face_detection.FaceDetection() as fd:
    with hands.Hands() as hand_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = fd.process(frame_rgb)  # type: ignore
            hand_results = hand_detection.process(frame_rgb)  # type: ignore

            if face_results.detections:
                for detection in face_results.detections:
                    drawing_utils.draw_detection(frame, detection)

            if hand_results.multi_hand_landmarks:
                for hand_landmark in hand_results.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(frame, hand_landmark)

            if face_results.detections and hand_results.multi_hand_landmarks:
                for detection in face_results.detections:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        landmark = hand_landmarks.landmark[8]  # just the index finger
                        face_box = detection.location_data.relative_bounding_box
                        if (
                            face_box.xmin < landmark.x
                            and face_box.xmin + face_box.width > landmark.x
                            and face_box.ymin < landmark.y
                            and face_box.ymin + face_box.height > landmark.y
                        ):
                            print("touching", i)
                            i += 1

            cv2.imshow("Hands and Face detection", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
