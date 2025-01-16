import face_recognition
import cv2
import os

# Lade die bekannten Gesichter
def load_known_faces(data_path):
    """
    Lädt bekannte Gesichter aus dem angegebenen Verzeichnis und erstellt Face Encodings.

    Parameter:
        data_path (str): Pfad zum Verzeichnis mit Trainingsbildern.

    Rückgabewerte:
        known_encodings (list): Liste der Face Encodings für bekannte Gesichter.
        known_names (list): Liste der Namen, die den Encodings zugeordnet sind.
    """
    known_encodings = []  # Speichert die Gesichtseigenschaften (Encodings)
    known_names = []      # Speichert die Namen zu den Encodings
    
    for file_name in os.listdir(data_path):  # Gehe durch alle Dateien im Datenverzeichnis
        name = os.path.splitext(file_name)[0].split(".")[0]  # Extrahiere den Namen aus dem Dateinamen
        file_path = os.path.join(data_path, file_name)       # Erstelle den vollständigen Pfad
        
        image = face_recognition.load_image_file(file_path)  # Lade das Bild
        encoding = face_recognition.face_encodings(image)[0]  # Extrahiere das erste Gesicht-Encoding (erster Wert der Liste)
        
        known_encodings.append(encoding)  # Füge das Encoding zur Liste hinzu
        known_names.append(name)          # Füge den Namen zur Liste hinzu
    
    return known_encodings, known_names

# Trainingsdaten laden
data_path = "./data_FaceID"  # Verzeichnis mit Trainingsbildern
known_encodings, known_names = load_known_faces(data_path)  # Lade bekannte Gesichter

# Webcam starten
cap = cv2.VideoCapture(0)  # Starte die Webcam
print("Face recognition mit face_recognition startet... Drücke 'q', um zu beenden.")

while True:
    ret, frame = cap.read()  # Lese ein Frame von der Webcam
    if not ret:
        break  # Beende die Schleife, falls kein Frame gelesen werden konnte
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konvertiere das Bild von BGR (OpenCV-Format) zu RGB (face_recognition-Format)
    face_locations = face_recognition.face_locations(rgb_frame)  # Finde die Positionen aller Gesichter im Frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # Generiere Encodings für die erkannten Gesichter
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):  
        # Für jedes erkannte Gesicht: 
        # Die Position (top, right, bottom, left) und das Encoding face_encoding
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)  
        # Vergleiche das Encoding mit den bekannten Encodings (tolerance = Genauigkeit)
        #Höhere Toleranz (z. B. 0.7) → Mehr Treffer, aber höhere Fehlerrate.
        #Niedrigere Toleranz (z. B. 0.4) → Weniger Treffer, aber präzisere Ergebnisse.
        
        name = "Unbekannt"  # Standardmäßig wird "Unbekannt" gesetzt
        
        if True in matches:  # Falls eine Übereinstimmung gefunden wurde
            match_index = matches.index(True)  # Hole den Index der Übereinstimmung
            name = known_names[match_index]   # Hole den Namen, der zu diesem Encoding gehört
        
        # Zeichne ein Rechteck um das erkannte Gesicht
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Zeichne den Namen über das Gesicht
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Zeige den aktuellen Frame im Fenster "Face Recognition"
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Falls die Taste 'q' gedrückt wird, beende die Schleife
        break

# Beende die Webcam und schließe alle Fenster
cap.release()
cv2.destroyAllWindows()
