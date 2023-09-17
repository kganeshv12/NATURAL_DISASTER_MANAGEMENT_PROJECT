import cv2
import numpy as np
import pygame
import tensorflow as tf
import time


model = tf.keras.models.load_model('cnn_1.h5')
pygame.mixer.init()

def play_audio():
    pygame.mixer.music.load("beep.mp3")
    pygame.mixer.music.play()


cap = cv2.VideoCapture(0)  


fire_detected = False

while True:
    
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (64, 64))

    
    frame_preprocessed = frame_resized / 255.0
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

    
    prediction = model.predict(frame_preprocessed)

    
    if prediction[0][0] > 0.5:
        fire_detected = False
    else:
        fire_detected = True

    # SOUND
    if fire_detected:
        play_audio()
        break
    else:
        pygame.mixer.music.stop()

    cv2.imshow('Fire Detection', frame)
    # print("FIRE DETECTED!!!")
    # BREAK
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# play_audio()

cap.release()
cv2.destroyAllWindows()
play_audio()

