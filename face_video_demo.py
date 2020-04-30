
'''
demo for camera
'''

import numpy as np
import cv2
import face_recognition
from face import Face
from utils import putText
from utils import preprocess_input

model = Face(train=False)
model.load_weights('./face_weights/face_weights.26-val_loss-3.85-val_age_loss-3.08-val_gender_loss-0.22-val_race_loss-0.55.utk.h5')

gender_labels = ['Male', 'Female']
race_labels = ['Whites', 'Blacks', 'Asian', 'Indian', 'Others']
#https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
age_labels = np.reshape(np.arange(1, 94), (93,1))

cap = cv2.VideoCapture(0)

while cap.isOpened:
    ret, frame = cap.read()

    face_locations = face_recognition.face_locations(frame, model='hog')

    if len(face_locations) > 0:
        face_batch = np.empty((len(face_locations), 200, 200, 3))

        # add face images into batch
        for i,rect in enumerate(face_locations):
            face_img = frame[rect[0]:rect[2], rect[3]:rect[1], :]
            face_img = cv2.resize(face_img, (200, 200))
            face_batch[i, :, :, :] = face_img
        
        face_batch = preprocess_input(face_batch)
        preds = model.predict(face_batch)

        preds_ages = preds[0]
        preds_genders = preds[1]
        preds_races = preds[2]

        # dispaly on srceen
        for rect, age, gender, race in zip(face_locations, preds_ages, preds_genders, preds_races):
            cv2.rectangle(frame, (rect[3], rect[0]), (rect[1], rect[2]), (255, 0, 0), 2)
            age = np.expand_dims(age, 0)
            # https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
            age_data = int(age.dot(age_labels).flatten())
            gender_index = np.argmax(gender)
            race_index = np.argmax(race)
            frame = putText(frame, 'gender: {0}'.format(gender_labels[gender_index]), (255, 0, 0), (rect[3], rect[0]-16), size=15)
            frame = putText(frame, 'race: {0}'.format(race_labels[race_index]), (255, 0, 0), (rect[3], rect[0]-32), size=15)
            frame = putText(frame, 'age: {0}'.format(age_data), (255, 0, 0), (rect[3], rect[0]-48), size=15)
    
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

