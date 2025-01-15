import cv2
import mediapipe as mp
import numpy as np
import torch
import utils
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import os


# Network Class
class GestureClassifierNet(nn.Module):

    dataset_path = "./dataset/gesture_dataset_new.csv"
    net_path = "./models/gesture_classification_model.pth"


    net = None

    def __init__(self):
        super(GestureClassifierNet, self).__init__()
        self.fc1 = nn.Linear(21 * 3, 128)
        self.fc2 = nn.Linear(128, 6)
    

    def forward(self, x):
        x = x.view(-1, 21 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    def train_model(self, dataset, num_epochs, optimizer, criterion):
        # Modell Training
        
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(data.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss}")
        
    def save_model(self, file_path):
        # Save the modell as pth
        torch.save(self.state_dict(), file_path)
        print(f'Modell wurde unter {file_path} gespeichert.')


    def train_or_load_net(self):

        # train network with dataset ot load if existing

        #create dataset
        data, labels = utils.load_data(self.dataset_path)
        normalized_data = utils.normalize_landmarks(data)

        x_train, y_train = torch.tensor(normalized_data, dtype = torch.float32), torch.tensor(labels, dtype = torch.long)
        dataset = TensorDataset(x_train, y_train)

        # Definition criterion, optimizer
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)


        if os.path.exists(self.net_path):
            # Load the trained model
            self.load_state_dict(torch.load(self.net_path))
            print(f"Trained model loaded from {self.net_path}")
        else:
            self.train_model(dataset, 10, optimizer, criterion)
            self.save_model(self.net_path)
        




class GestureRecognition:

    labels_map = {
        
        0: "Open",          #🖐️
        1: "Close",         #✊
        2: "Thumbs Up",     #👍
        3: "OK",            #👌
        4: "Swag",           #🤙
        5: "finger up"

    }    


    def __init__(self):
        # Stelle sicher, dass mp.solutions.hands richtig initialisiert wird
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2)

        # Neue Variablen für die "Pointer"-Geste
        self.pointer_count = 0
        self.pointer_start_time = 0
        self.pointer_hold_threshold = 3  # Schwelle in Sekunden


    def draw_hand_points(self, landmark, frame):
        # Get landmark position
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])    
        # Draw landmark
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)    
        
    def draw_lines(self, landmarks, frame):
            # Connect landmarks to form a hand representation
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                    (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (0, 5), (0, 17), (5,9), (9, 13), (13, 17), (0, 9), (0, 13)]

        for connection in connections:
            pt1 = (int(landmarks[connection[0]][0] * frame.shape[1]), int(landmarks[connection[0]][1] * frame.shape[0]))
            pt2 = (int(landmarks[connection[1]][0] * frame.shape[1]), int(landmarks[connection[1]][1] * frame.shape[0]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  
        
    def convert_landmarks(self, landmarks):
        
        mean = np.mean(landmarks, axis=0)
        std = np.std(landmarks, axis=0)
        normalized_landmarks = (landmarks - mean) / std    
            
        landmark_tensor = torch.tensor(normalized_landmarks, dtype=torch.float32) 
        return landmark_tensor

    def predict_label_from_landmarks(self, hand_landmarks, frame, net):

        
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark])

        if landmarks is not None:
            landmark_tensor = self.convert_landmarks(landmarks)


        x = int(landmarks[0][0] * frame.shape[1])
        y = int(landmarks[0][1] * frame.shape[0])
        
        output = net(landmark_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Bestimme die vorhergesagte Klasse (Index mit der höchsten Wahrscheinlichkeit)
        _, predicted_class = torch.max(probabilities, 1)

        predicted_label = self.labels_map[predicted_class.item()]
        cv2.putText(frame, predicted_label, (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return predicted_label





    def visualize_hand(self, hand_landmarks, frame):

        #draw landmarks
        for landmark in hand_landmarks.landmark:
            self.draw_hand_points(landmark, frame)
                    
                #self.draw_lines(landmarks, frame)


