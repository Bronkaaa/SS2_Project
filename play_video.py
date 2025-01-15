import cv2

# Videodatei laden
video_path = './videos/video20241219_210045.avi'
cap = cv2.VideoCapture(video_path)

# Überprüfen, ob das Video geöffnet wurde
if not cap.isOpened():
    print("Fehler beim Öffnen des Videos")
    exit()

# Hole die Frame-Rate des Videos (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / 35)  # Berechne die Verzögerung in Millisekunden pro Frame

# Video abspielen
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Beende die Schleife, wenn das Video zu Ende ist

    # Zeige das aktuelle Frame an
    cv2.imshow('Video', frame)

    # Warte die richtige Zeit, um die Framerate zu simulieren und auf 'q' drücken, um das Video zu stoppen
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
