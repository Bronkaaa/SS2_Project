import numpy as np
import csv

    
    
def normalize_landmarks(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data


# 1. Daten laden
def load_data(csv_file):
    data = []
    labels = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(int(row[0]))
            landmarks = np.array(row[1:], dtype=np.float32)
            data.append(landmarks)
    return np.array(data), np.array(labels)
