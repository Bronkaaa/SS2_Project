import cv2
import mediapipe as mp


class FaceRecognition:
    def __init__(self):
        # Initialisiere Mediapipe für Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)

    def visualize_face_landmarks(self, results, frame):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


    def get_face_id():
        pass



    def get_middle_point(self, results, frame):

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            # Greife direkt auf das erste Gesicht zu
            first_face_landmarks = results.multi_face_landmarks[0]
            x = int((first_face_landmarks.landmark[1].x + first_face_landmarks.landmark[2].x) * w / 2)
            y = int((first_face_landmarks.landmark[1].y + first_face_landmarks.landmark[2].y) * h / 2)
            z = (first_face_landmarks.landmark[1].z + first_face_landmarks.landmark[2].z) / 2
            face_center = (x, y, z)
            
            return face_center  # Gibt den Mittelpunkt des ersten Gesichts zurück
        
        return None
    




