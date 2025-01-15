from onvif import ONVIFCamera
import time
import threading
import cv2

class OnvifCamera:
    def __init__(self, ip_address, port, username, password):
        self.camera = ONVIFCamera(ip_address, port, username, password)
        self.username = username
        self.password = password

        self.ptz_service = self.camera.create_ptz_service()
        self.media_service = self.camera.create_media_service()

        # Hole das erste Profil, um sicherzustellen, dass es verfügbar ist
        profiles = self.media_service.GetProfiles()
        self.profile = profiles[0]  # Wähle das erste Profil aus

        self.zoom_mode = False
        self.MOVE_MODE = {
            0: "SEARCH_FACE",
            1: "FOLLOW_FACE",
            2: "FOLLOW_FINGER",
            3: "FOCUS",
            4: "FOCUS_BLACKBOARD",
            5: "FOCUS_AND_EQUALIZE"
        }
        self.current_mode = self.MOVE_MODE[0]

        self.max_zoom_out()


    def get_device_info(self):
        device_info = self.camera.devicemgmt.GetDeviceInformation()
        return {
            "Hersteller": device_info.Manufacturer,
            "Modell": device_info.Model,
            "Firmware-Version": device_info.FirmwareVersion
        }

    def get_stream_uri(self):
        """Gibt die Stream-URL der Kamera zurück"""
        stream_uri = self.media_service.GetStreamUri({
            'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
            'ProfileToken': self.profile.token
        })
        return stream_uri.Uri

    def get_auth_stream_url(self, stream_url):
        return stream_url.replace("rtsp://", f"rtsp://{self.username}:{self.password}@")

    def move(self, pan_speed=0, tilt_speed=0, zoom_speed=0):
        """Bewege die Kamera mit den angegebenen Geschwindigkeiten"""
        request = self.ptz_service.create_type('ContinuousMove')
        request.ProfileToken = self.profile.token
        request.Velocity = {
            'PanTilt': {'x': pan_speed, 'y': -tilt_speed},
            'Zoom': {'x': zoom_speed}
        }
        self.ptz_service.ContinuousMove(request)



    def stop(self):
        """Stoppt alle Bewegungen"""
        request = self.ptz_service.create_type('Stop')
        request.ProfileToken = self.profile.token
        self.ptz_service.Stop(request)

    def pan_left(self, speed=0.5, duration=2):
        print("Schwenke nach links")
        self.move(pan_speed=-speed)
        self.stop()

    def pan_right(self, speed=0.5, duration=2):
        print("Schwenke nach rechts")
        self.move(pan_speed=speed)
        self.stop()

    def tilt_up(self, speed=0.5, duration=2):
        print("Neige nach oben")
        self.move(tilt_speed=speed)
        self.stop()

    def tilt_down(self, speed=0.5, duration=2):
        print("Neige nach unten")
        self.move(tilt_speed=-speed)


    def zoom_in(self, speed, duration):
        """Zoomt in, indem es die Bewegung in einem eigenen Thread ausführt"""
        def zoom_in_thread():
            self.move(zoom_speed=speed)
            time.sleep(duration)
            self.stop()

        threading.Thread(target=zoom_in_thread, daemon=True).start()  # Der Thread läuft im Hintergrund


    def zoom_out(self, speed, duration):
        """Zoomt heraus, indem es die Bewegung in einem eigenen Thread ausführt"""
        def zoom_out_thread():
            self.move(zoom_speed=-speed)
            time.sleep(duration)
            self.stop()

        threading.Thread(target=zoom_out_thread, daemon=True).start()  # Der Thread läuft im Hintergrund


    def max_zoom_out(self):
        """Setze die Kamera auf den maximalen Zoom-Out-Wert"""

        self.zoom_out(1, 1)  # Hier wird die maximale Geschwindigkeit für den Zoom-Out eingestellt
        time.sleep(3)  # 5 Sekunden herauszoomen, je nach Bedarf anpassen
        self.stop()


    def track_face(self, face_center, frame, threshold=10, move_threshold=0.1, max_speed=0.6, min_speed=0.3):


        """Bewege die Kamera, um das Gesicht zu verfolgen."""
        h, w, _ = frame.shape

        # Berechne die Differenz zwischen dem Mittelpunkt des Gesichts und dem Mittelpunkt des Bildes
        x_diff = face_center[0] - w // 2
        y_diff = face_center[1] - h // 2
        z_diff = face_center[2]  # Verwende z-Wert zur Bestimmung der Entfernung

        # Berechne die Pan- und Tilt-Geschwindigkeiten basierend auf der Differenz
        pan_speed = 0.5 * (x_diff / (w // 2))  # Pan-Geschwindigkeit basierend auf der horizontalen Position
        tilt_speed = 0.5 * (y_diff / (h // 2))  # Tilt-Geschwindigkeit basierend auf der vertikalen Position

        # Begrenze die Geschwindigkeit, um zu schnelle Bewegungen zu verhindern
        pan_speed = max(-max_speed, min(max_speed, pan_speed))
        tilt_speed = max(-max_speed, min(max_speed, tilt_speed))

        # Wenn Pan- oder Tilt-Geschwindigkeit unter dem move_threshold sind, setze sie auf 0
        if abs(pan_speed) < move_threshold:
            pan_speed = 0
        if abs(tilt_speed) < move_threshold:
            tilt_speed = 0

        # Setze eine Mindestgeschwindigkeit für Pan und Tilt
        if pan_speed != 0 and abs(pan_speed) < min_speed:
            pan_speed = min_speed * (1 if pan_speed > 0 else -1)

        if tilt_speed != 0 and abs(tilt_speed) < min_speed:
            tilt_speed = min_speed * (1 if tilt_speed > 0 else -1)

        # Wenn eine signifikante Bewegung erforderlich ist, führe die Kamera-Bewegung aus
        if abs(pan_speed) > 0 or abs(tilt_speed) > 0:
            # Nur wenn die Differenz über dem Schwellenwert liegt oder eine signifikante Bewegung erforderlich ist

            if abs(x_diff) > threshold or abs(y_diff) > threshold:

                #negatives vorzeichen da kamera kopfüber. später ändern
                self.move(-pan_speed, -tilt_speed)
        else:
            # Wenn keine Bewegung erforderlich ist, stoppe die Kamera
            self.stop()


    def track_finger(self, finger_tip, frame, threshold=30, move_threshold=0.1, max_speed=0.7, min_speed=0.3):
        """Bewege die Kamera, um den Finger zu verfolgen (mit Landmarke des Zeigefingers)."""
        h, w, _ = frame.shape

        # Extrahiere die x, y Koordinaten des Zeigefingerspitzes
        x_finger, y_finger = int(finger_tip.x * w), int(finger_tip.y * h)

        # Berechne die Differenz zwischen dem Mittelpunkt des Zeigefingers und dem Mittelpunkt des Bildes
        x_diff = x_finger - w // 2
        y_diff = y_finger - h // 2

        # Berechne die Pan- und Tilt-Geschwindigkeiten basierend auf der Differenz
        pan_speed = 0.5 * (x_diff / (w // 2))  # Pan-Geschwindigkeit basierend auf der horizontalen Position
        tilt_speed = 0.5 * (y_diff / (h // 2))  # Tilt-Geschwindigkeit basierend auf der vertikalen Position

        # Begrenze die Geschwindigkeit, um zu schnelle Bewegungen zu verhindern
        pan_speed = max(-max_speed, min(max_speed, pan_speed))
        tilt_speed = max(-max_speed, min(max_speed, tilt_speed))

        # Wenn Pan- oder Tilt-Geschwindigkeit unter dem move_threshold sind, setze sie auf 0
        if abs(pan_speed) < move_threshold:
            pan_speed = 0
        if abs(tilt_speed) < move_threshold:
            tilt_speed = 0

        # Setze eine Mindestgeschwindigkeit für Pan und Tilt
        if pan_speed != 0 and abs(pan_speed) < min_speed:
            pan_speed = min_speed * (1 if pan_speed > 0 else -1)

        if tilt_speed != 0 and abs(tilt_speed) < min_speed:
            tilt_speed = min_speed * (1 if tilt_speed > 0 else -1)

        # Wenn eine signifikante Bewegung erforderlich ist, führe die Kamera-Bewegung aus
        if abs(pan_speed) > 0 or abs(tilt_speed) > 0:
            # Nur wenn die Differenz über dem Schwellenwert liegt oder eine signifikante Bewegung erforderlich ist
            if abs(x_diff) > threshold or abs(y_diff) > threshold:
                # Negatives Vorzeichen, da Kamera kopfüber sein könnte. Kann später angepasst werden.
                self.move(-pan_speed, -tilt_speed)
        else:
            # Wenn keine Bewegung erforderlich ist, stoppe die Kamera
            self.stop()


    def search_face(self):

        #self.max_zoom_out()
        self.move(0.5, 0)
        
            