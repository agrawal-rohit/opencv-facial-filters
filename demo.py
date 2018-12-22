# Execution

import numpy as np
from training import get_model, load_trained_model, compile_model

model = get_model()
compile_model(model)
load_trained_model(model)

import cv2

# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Get webcam
camera = cv2.VideoCapture(0)

while True:
    grab_trueorfalse, img = camera.read()
    
    # Preprocess input fram webcam
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        img_copy = np.copy(img)
        img_copy_1 = np.copy(img)
        roi_color = img_copy_1[y:y+h, x:x+w]
        
        width_original = roi_gray.shape[1]
        height_original = roi_gray.shape[0]
        img_gray = cv2.resize(roi_gray, (96, 96)) 
        img_gray = img_gray/255
        
        img_model = np.reshape(img_gray, (1,96,96,1))
        keypoints = model.predict(img_model)[0]
        
        x_coords = keypoints[0::2]
        y_coords = keypoints[1::2]
        
        x_coords_normalized = (x_coords+0.5)*width_original
        y_coords_normalized = (y_coords+0.5)*height_original
        
        for i in range(len(x_coords)):  
            cv2.circle(roi_color, (x_coords_normalized[i], y_coords_normalized[i]), 2, (255,255,0), -1)
        
        
        left_lip_coords = (int(x_coords_normalized[11]), int(y_coords_normalized[11]))
        right_lip_coords = (int(x_coords_normalized[12]), int(y_coords_normalized[12]))
        top_lip_coords = (int(x_coords_normalized[13]), int(y_coords_normalized[13]))
        bottom_lip_coords = (int(x_coords_normalized[14]), int(y_coords_normalized[14]))
        
        left_eye_coords = (int(x_coords_normalized[3]), int(y_coords_normalized[3]))
        right_eye_coords = (int(x_coords_normalized[5]), int(y_coords_normalized[5]))
        brow_coords = (int(x_coords_normalized[6]), int(y_coords_normalized[6]))
        
        beard_width = right_lip_coords[0] - left_lip_coords[0]
        glasses_width = right_eye_coords[0] - left_eye_coords[0]
        
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

        # Beard filter
        santa_filter = cv2.imread('filters/santa_filter.png', -1)
        santa_filter = cv2.resize(santa_filter, (beard_width*3,150))
        sw,sh,sc = santa_filter.shape
        
        for i in range(0,sw):
            for j in range(0,sh):
                if santa_filter[i,j][3] != 0:
                    img_copy[top_lip_coords[1]+i+y-20, left_lip_coords[0]+j+x-60] = santa_filter[i,j]
                    
        # Hat filter
        hat = cv2.imread('filters/hat2.png', -1)
        hat = cv2.resize(hat, (w,w))
        hw,hh,hc = hat.shape
        
        for i in range(0,hw):
            for j in range(0,hh):
                if hat[i,j][3] != 0:
                    img_copy[i+y-brow_coords[1]*2, j+x-left_eye_coords[0]*1 + 20] = hat[i,j]
        
        # Glasses filter
        glasses = cv2.imread('filters/glasses.png', -1)
        glasses = cv2.resize(glasses, (glasses_width*2,150))
        gw,gh,gc = glasses.shape
        
        for i in range(0,gw):
            for j in range(0,gh):
                if glasses[i,j][3] != 0:
                    img_copy[brow_coords[1]+i+y-50, left_eye_coords[0]+j+x-60] = glasses[i,j]
        
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)
        cv2.imshow('Output',img_copy)
        cv2.imshow('Keypoints predicted',img_copy_1)
    cv2.imshow('Webcam',img)
    
    
    if cv2.waitKey(1) & 0xFF == ord("e"):   # If 'e' is pressed, stop reading
        break