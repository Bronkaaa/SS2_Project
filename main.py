import cv2
from Gesture_Recognition import GestureRecognition, GestureClassifierNet
from Face_Recognition import FaceRecognition
import time 
from CameraControl import OnvifCamera
import numpy as np
import math
import os
import keyboard
import queue
import datetime

class CameraStream:
    def __init__(self, stream_url, camera):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        self.camera = camera

        # Initialize VideoWriter (for saving the video stream)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = f"./videos/video{timestamp}.avi"
        self.fps = 30  # Set a default FPS, you can modify it
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'H264'
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.is_recording = False


        #gesture recognition
        self.gesture_classifier = GestureClassifierNet()
        self.gesture_recognition = GestureRecognition()

        self.gesture_data = {
            'hand_1': {
                'gesture_hold_duration' : 2.5,
                'last_hand_detection_time' : None,
                'gesture_finger_up': None,
                'gesture_close': None,
                'gesture_open': None,
                'gesture_ok': None,
                'thumbs_up': None,
                'gesture_ok_hold': False,
                'last_landmark': None,
                'previous_landmark': None,
                'frame_counter': 0,  # Zähler für Frames
            },
            'hand_2': {
                'gesture_hold_duration' : 2.5,
                'last_hand_detection_time' : None,
                'gesture_finger_up': None,
                'gesture_close': None,
                'gesture_open': None,
                'gesture_ok': None,
                'thumbs_up': None,
                'gesture_ok_hold': False,
                'last_landmark': None,
                'previous_landmark': None,
                'frame_counter': 0,  # Zähler für Frames
            }
        }

        #face recognition
        self.last_face_detection_time = None
        self.face_recognition = FaceRecognition()
        self.frame_counter = 0

        self.is_zooming = False  # Flag für den Zoom-Prozess
        self.zoom_thread = None  # Der Thread für den Zoom-Prozess
        self.frame_queue = queue.Queue()  # Queue für den sicheren Austausch der Frames und Face-Landmarks
        self.last_face_width_percent = None  # Speichert die letzte berechnete Gesichtsbreite
        self.stop_thread = False  # Flag zum Stoppen des Threads

        self.apply_transformation = False  # Flag to apply the transformation
        self.points = []  # To store the four points


    def start_stream(self):

        if not self.cap.isOpened():
            print("Konnte den Stream nicht öffnen.")
            return
        
        self.gesture_classifier.train_or_load_net()
        self.gesture_classifier.eval()  # Setze das Modell in den Evaluationsmodus 

        #fps calculation
        start_time = time.time()
        frame_count = 0
        fps = 0


        cv2.namedWindow('IMG')
        cv2.setMouseCallback('IMG', self.select_points)


        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Konnte den Stream nicht abrufen.")
                break

            # Drehe das Bild um 180 Grad
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            #optinal test
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Geometrische Entzerrung anwenden, falls aktiviert
            if self.apply_transformation and len(self.points) == 4:
                pts_src = np.array(self.points, dtype="float32")
                width = 400  # Breite des transformierten Bildes
                height = 300  # Höhe des transformierten Bildes
                pts_dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")

                # Perspektivtransformation berechnen
                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                frame = cv2.warpPerspective(frame, matrix, (width, height))



            else:
                # gesture_recognition
                results_hands = self.gesture_recognition.hands.process(rgb_frame)
                if results_hands.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):

                        hand_key = 'hand_1' if idx == 0 else 'hand_2'
                        # Aktualisiere die Zeit der letzten Handerkennung
                        self.gesture_data[hand_key]['last_hand_detection_time'] = time.time()

                        self.gesture_recognition.visualize_hand(hand_landmarks, frame)

                        predicted_label = self.gesture_recognition.predict_label_from_landmarks(hand_landmarks, frame, self.gesture_classifier)

                        self.set_current_cam_mode(predicted_label, hand_key, hand_landmarks)
                    

                
                # Gesichtserkennung mit detaillierten Gesichtspunkten

                results_face = self.face_recognition.face_mesh.process(rgb_frame)            
              
                if results_face.multi_face_landmarks and self.camera.current_mode != "FOCUS_AND_EQUALIZE":
                    self.face_recognition.visualize_face_landmarks(results_face, frame)
                    self.last_face_detection_time = time.time()

                


                # andere methode starten je nach aktuellem mode() track_face, track_finger...)
                ## diesen block in seperate methode packen...
                if results_hands or results_face:
                    self.change_cam_mode(frame, results_face, results_hands)
                    self.camera_zoom()



            # Increment frame count
            self.frame_counter += 1
            frame_count += 1
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            # If one second has elapsed, calculate FPS
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # show image

         # Punkte anzeigen
            for point in self.points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)


            self.out.write(frame)

            cv2.imshow('IMG', frame)

            key = cv2.waitKey(1) & 0xFF  # Store key press result
            if key == ord('q'):
                self.release()
            elif key == ord('c'):  # Punkte zurücksetzen
                self.points = []
                self.apply_transformation = False
                self.camera.current_mode = "FOCUS"
                print("Punkte zurückgesetzt.")
            elif key == ord('a') and len(self.points) == 4:  # Transformation anwenden
                self.apply_transformation = True
                self.camera.current_mode = "FOCUS_AND_EQUALIZE"
                print("Transformation angewendet.")
            elif key == ord('s'):  # Punkte zurücksetzen
                # Zeitstempel für eindeutigen Dateinamen
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"./images/capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Bild wurde gespeichert: {filename}")
            elif keyboard.is_pressed('z'):
                self.camera.zoom_in()



    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point selected: {x}, {y}")



    def release(self):
        self.cap.release()
        self.out.release()  # Ensure the video file is saved properly
        cv2.destroyAllWindows()


    def set_current_cam_mode(self, predicted_label, hand_key, hand_landmarks):
        # Geste 1 (Finger hoch)
        if predicted_label == 'finger up': 
            if self.gesture_data[hand_key]['gesture_finger_up'] is None:
                self.gesture_data[hand_key]['gesture_finger_up'] = time.time()
            else:
                if time.time() - self.gesture_data[hand_key]['gesture_finger_up'] >= self.gesture_data[hand_key]['gesture_hold_duration']:
                    self.gesture_data[hand_key]['gesture_finger_up'] = None  # Zurücksetzen, wenn Geste nicht mehr erkannt
                    self.camera.current_mode = "FOLLOW_FINGER"

        # Geste 2 (Schließen)
        elif predicted_label == "Close" and (self.camera.current_mode == "FOLLOW_FINGER" or self.camera.current_mode == "FOLLOW_FACE"):
            if self.gesture_data[hand_key]['gesture_close'] is None:
                self.gesture_data[hand_key]['gesture_close'] = time.time()
            else:
                if time.time() - self.gesture_data[hand_key]['gesture_close'] >= self.gesture_data[hand_key]['gesture_hold_duration']:
                    self.gesture_data[hand_key]['gesture_close'] = None  # Zurücksetzen, wenn Geste nicht mehr erkannt
                    self.camera.current_mode = "FOCUS"

        # Geste 3 (Öffnen)
        elif predicted_label == "Open" and self.camera.current_mode == "FOCUS":
            if self.gesture_data[hand_key]['gesture_open'] is None:
                self.gesture_data[hand_key]['gesture_open'] = time.time()
            else:
                if time.time() - self.gesture_data[hand_key]['gesture_open'] >= self.gesture_data[hand_key]['gesture_hold_duration']:
                    self.gesture_data[hand_key]['gesture_open'] = None
                    self.camera.current_mode = "FOLLOW_FACE"

            # Geste 3 (Öffnen)
        elif predicted_label == "Thumbs Up" and self.camera.current_mode == "FOCUS":
            if self.gesture_data[hand_key]['thumbs_up'] is None:
                self.gesture_data[hand_key]['thumbs_up'] = time.time()
            else:
                if time.time() - self.gesture_data[hand_key]['thumbs_up'] >= self.gesture_data[hand_key]['gesture_hold_duration']:
                    self.gesture_data[hand_key]['thumbs_up'] = None
                    self.camera.current_mode = "FOCUS_BLACKBOARD"

        elif predicted_label == "OK":
            if self.gesture_data[hand_key]['gesture_ok'] is None:
                self.gesture_data[hand_key]['gesture_ok'] = time.time()
            else:
                # Prüfe, ob die "OK"-Geste für die erforderliche Dauer gehalten wurde
                if time.time() - self.gesture_data[hand_key]['gesture_ok'] >= self.gesture_data[hand_key]['gesture_hold_duration']:
                    self.gesture_data[hand_key]['gesture_ok'] = None
                    self.gesture_data[hand_key]['gesture_ok_hold'] = True
        else:
            # Zurücksetzen, wenn keine der Gesten erkannt wurde
            self.gesture_data[hand_key]['gesture_finger_up'] = None
            self.gesture_data[hand_key]['gesture_close'] = None
            self.gesture_data[hand_key]['gesture_open'] = None
            self.gesture_data[hand_key]['gesture_ok'] = None
            self.gesture_data[hand_key]['gesture_ok_hold'] = False


        self.gesture_data[hand_key]['frame_counter'] += 1
        if self.gesture_data[hand_key]['frame_counter'] % 1 == 0:
            self.gesture_data[hand_key]['previous_landmark'] = self.gesture_data[hand_key]['last_landmark']
            self.gesture_data[hand_key]['last_landmark'] = hand_landmarks.landmark[8]  # Zeigefingerspitze



    def camera_zoom(self):

        zoom_threshhold = 0.01
        
        if self.gesture_data['hand_1']['gesture_ok_hold'] and self.gesture_data['hand_2']['gesture_ok_hold'] and self.camera.current_mode == "FOLLOW_FACE":
            # Berechne die Distanz zwischen den Zeigefingerspitzen der beiden Hände
            if self.gesture_data['hand_1']['last_landmark'] and self.gesture_data['hand_2']['last_landmark']:

                finger_1_tip = self.gesture_data['hand_1']['last_landmark']
                finger_2_tip = self.gesture_data['hand_2']['last_landmark']

                # Extrahiere die 3D-Koordinaten der Zeigefingerspitzens
                tip_1 = (finger_1_tip.x, finger_1_tip.y, finger_1_tip.z)
                tip_2 = (finger_2_tip.x, finger_2_tip.y, finger_2_tip.z)

                # Berechne die Distanz im aktuellen Frame
                distance = self.calculate_distance(tip_1, tip_2)


                # Berechne die Distanz der vorherigen Positionen
                if self.gesture_data['hand_1']['previous_landmark'] and self.gesture_data['hand_2']['previous_landmark']:
                    prev_finger_1_tip = self.gesture_data['hand_1']['previous_landmark']
                    prev_finger_2_tip = self.gesture_data['hand_2']['previous_landmark']

                    prev_tip_1 = (prev_finger_1_tip.x, prev_finger_1_tip.y, prev_finger_1_tip.z)
                    prev_tip_2 = (prev_finger_2_tip.x, prev_finger_2_tip.y, prev_finger_2_tip.z)

                    prev_distance = self.calculate_distance(prev_tip_1, prev_tip_2)

                    # Wenn der Abstand größer ist als der Schwellenwert, führe den Zoom aus
                    if abs(distance - prev_distance) > zoom_threshhold:
                        # Vergleiche den Abstand im aktuellen Frame mit dem vorherigen Frame
                        distance_change = distance - prev_distance

                        # Wenn sich die Hände nähern
                        if distance_change > 0:
                            # Berechne die Zoomstärke proportional zur Veränderung der Distanz
                            zoom_factor = abs(distance_change) * 10  # Der Faktor '10' kann angepasst werden
                            self.camera.zoom_in(1,1)

                        # Wenn sich die Hände entfernen
                        elif distance_change < 0:
                            # Berechne die Zoomstärke proportional zur Veränderung der Distanz
                            zoom_factor = distance_change * 10  # Der Faktor '10' kann angepasst werden
                            self.camera.zoom_out(1,1)



    def calculate_distance(self, tip_1, tip_2):
        # Calculate the Euclidean distance between two finger tips in 3D space
        x1, y1, z1 = tip_1
        x2, y2, z2 = tip_2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    # anpassen auf neues dict
    # evtl ändern, damit die eine hand nicht die andere überschreibt und es zu komischen aktionen kommt.
    def change_cam_mode(self, frame, results_face, results_hands):
            # andere methode starten je nach aktuellem mode()track_face, track_finger...)
        ## diesen block in seperate methode packen...
        if self.camera.current_mode == "SEARCH_FACE":
            if not results_face.multi_face_landmarks:
                self.camera.search_face()
            else:
                self.camera.current_mode = "FOLLOW_FACE"

        elif self.camera.current_mode == "FOLLOW_FACE":
            if results_face.multi_face_landmarks:
                face_center = self.face_recognition.get_middle_point(results_face, frame)

                if self.frame_counter % 5 == 0:
                    self.camera.track_face(face_center, frame)


            #falls 10 sek kein gesicht mehr gefunden wurde, suche gesicht
            else:
                if time.time() - self.last_face_detection_time > 10:
                    self.camera.current_mode = "SEARCH_FACE"
                    self.last_face_detection_time = None

        elif self.camera.current_mode == "FOLLOW_FINGER":
            if results_hands.multi_hand_landmarks:
                finger_tip = results_hands.multi_hand_landmarks[0].landmark[8]
                if self.frame_counter % 5 == 0:
                    self.camera.track_finger(finger_tip, frame)
            else:
                if self.gesture_data['hand_1']['last_hand_detection_time'] is not None:
                    if (time.time() - self.gesture_data['hand_1']['last_hand_detection_time'] > 10): 
                        self.camera.current_mode = "SEARCH_FACE"
                        self.gesture_data['hand_1']['last_hand_detection_time'] = None
                if self.gesture_data['hand_2']['last_hand_detection_time'] is not None:
                    if (time.time() - self.gesture_data['hand_2']['last_hand_detection_time'] > 10): 
                        self.camera.current_mode = "SEARCH_FACE"
                        self.gesture_data['hand_2']['last_hand_detection_time'] = None

        elif self.camera.current_mode == "FOCUS":
            self.camera.stop()

        elif self.camera.current_mode == "FOCUS_AND_EQUALIZE":
            self.camera.stop()



    # Funktion zum Speichern von Bildern in einem Ordner in einer einzigen Zeile
    def save_image(self, frame, folder_path, prefix="image"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # Ordner erstellen, falls er noch nicht existiert

        # Zeitstempel im Dateinamen
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(folder_path, f"{prefix}_{timestamp}.jpg"), frame)  # Bild speichern



# Initialisierung und Verwendung der Kamera
def main():
    camera = OnvifCamera('192.168.178.64', 80, 'admin', 'robuntu22')

    # print information
    device_info = camera.get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")
    stream_uri = camera.get_stream_uri()
    print("Stream URL:", stream_uri)




    auth_stream_url = camera.get_auth_stream_url(stream_uri)

    camera_stream = CameraStream(auth_stream_url, camera)

    try:   
        camera_stream.start_stream()
    finally:
        camera.stop()
        camera_stream.release()

if __name__ == "__main__":
    main()