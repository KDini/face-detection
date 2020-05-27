#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv

# Charger les classificateurs en cascade pré-entrainés
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')



# Charger les images 
# img = cv.imread('img/man1.jpeg')
img = cv.imread('img/group1.jpg')
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
    
    # Extraire les visages de l'image principale
    # OpenCV et Numpy: y <-> row et x<-> col
    face = img[y:y+h, x:x+w]
    
    # Afficher face0, face1, face2, etc ...
    cv.imshow('face{}'.format(i), face)
    i += 1
    
# Affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()