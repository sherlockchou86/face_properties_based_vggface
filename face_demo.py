

'''
demo for single image
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


demo_image = cv2.imread('./demo_images/how-old-demo5.jpg')
image_h, image_w = demo_image.shape[0], demo_image.shape[1]
margin = 0.01

face_locations = face_recognition.face_locations(demo_image, model='hog')

if len(face_locations) > 0:
    face_batch = np.empty((len(face_locations), 200, 200, 3))

    # add face images into batch
    for i,rect in enumerate(face_locations):
        # crop with a margin
        top, bottom, left, right = rect[0], rect[2], rect[3], rect[1]
        top = max(int(top - image_h * margin), 0)
        left = max(int(left - image_w * margin), 0)
        bottom = min(int(bottom + image_h * margin), image_h - 1)
        right = min(int(right + image_w * margin), image_w - 1)

        face_img = demo_image[top:bottom, left:right, :]
        face_img = cv2.resize(face_img, (200, 200))
        face_batch[i, :, :, :] = face_img
    
    face_batch = preprocess_input(face_batch)
    preds = model.predict(face_batch)

    preds_ages = preds[0]
    preds_genders = preds[1]
    preds_races = preds[2]

    # dispaly on srceen
    for rect, age, gender, race in zip(face_locations, preds_ages, preds_genders, preds_races):
        cv2.rectangle(demo_image, (rect[3], rect[0]), (rect[1], rect[2]), (255, 0, 0), 2)
        age = np.expand_dims(age, 0)
        # https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
        age_data = int(age.dot(age_labels).flatten())
        gender_index = np.argmax(gender)
        race_index = np.argmax(race)
        demo_image = putText(demo_image, 'gender: {0}'.format(gender_labels[gender_index]), (255, 0, 0), (rect[3], rect[0]-16), size=15)
        demo_image = putText(demo_image, 'race: {0}'.format(race_labels[race_index]), (255, 0, 0), (rect[3], rect[0]-32), size=15)
        demo_image = putText(demo_image, 'age: {0}'.format(age_data), (255, 0, 0), (rect[3], rect[0]-48), size=15)

    cv2.imshow('image', demo_image)
    if cv2.waitKey(0) & 0xff == ord("q"):
        cv2.destroyAllWindows()
