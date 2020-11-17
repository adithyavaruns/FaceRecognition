import face_recognition
import numpy as np
import time
import cv2
import glob, os
import csv
vec1 = []
start = time.time()
fold_cnt = 0
folders = glob.glob("testextract/*")
face_embeddings = []
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

with open("train_data.csv", 'w', newline='') as file_write:
    writer = csv.writer(file_write)
    for location in sorted(folders):
        for file_list in sorted(glob.glob(location + "\*.jpg")):
            images = cv2.imread(file_list)
            print("images", file_list)
            d = (sorted(glob.glob(location + "\*.jpg")))
            var = d[0].split('\\')[1]
            cls = var
            face_encodes = face_recognition.face_encodings(images)
            #print("face", face_encodes[0])
            #print("face_encodes", type(face_encodes), face_encodes[0])
            if not face_encodes:
                print("no embeds")
            else:
                vector = np.array(face_encodes[0])
                print("vector", vector.shape)
                file = np.append(cls, vector)
                writer.writerow(file)