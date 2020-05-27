#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv

# Charger les classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')



# Charger les images 
# img = cv.imread('img/woman1.jpeg')
img = cv.imread('img/woman2.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Executer de la dection facialede visage
# DetectMultiScale(image, scale factor, number of neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)



# Affichage des visages
i = 0
for face in faces:
    x, y, w,h = face
    
    # Dessiner le rectangle sur l'image principale
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    
# Exécution de la dectetion des yeux
eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)


# Affichage des yeux
for (ex, ey, ew, eh) in eyes:
    # Dessiner le rectangle autour des yeux sur l'image principale
    cv.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)
    
    
# Affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()