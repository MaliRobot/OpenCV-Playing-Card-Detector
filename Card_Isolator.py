# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 20:55:33 2018

@author: Misha
"""

### Takes a card picture and creates a top-down 200x300 flattened image
### of it. Isolates the suit and rank and saves the isolated images.
### Runs through A - K ranks and then the 4 suits.

# Import necessary packages
import cv2
import numpy as np
import time
import Cards
import os

img_path = os.path.dirname(os.path.abspath(__file__)) + '/Card_Imgs/'
debug_path = os.path.dirname(os.path.abspath(__file__)) + '/Debug_Imgs/'

IM_WIDTH = 800
IM_HEIGHT = 600 

RANK_WIDTH = 800
RANK_HEIGHT = 600

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# If using a USB Camera instead of a PiCamera, change PiOrUSB to 2
PiOrUSB = 2

debug_pics = 1

if PiOrUSB == 1:
    # Import packages from picamera library
    from picamera.array import PiRGBArray
    from picamera import PiCamera

    # Initialize PiCamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))

if PiOrUSB == 2:
    # Initialize USB camera
    cap = cv2.VideoCapture(0)

# Use counter variable to switch from isolating Rank to isolating Suit
i = 1

for Name in ['reito_lantern','ornate_kanzashi', 'free_from_the_real', 
              'sakura_tribe_scout', 'plains_ben_thomposon', 'path_of_angers_flame', 
              'sift_through_sands', 'setons_desire', 'phantom_nomad', 
              'divine_light', 'ghostly_wings', 'plains_fred_fields', 'locust_mister',
              'jugan_the_rising_star', 'whispering_shade', 'divergent_growth', 
              'ryusei_the_falling_star', 'dripping_tongue_zubera',
              'ninja_of_the _deep_hours', 'plains_matthew_mitchell', 'plains_greg_staples',
              'forest_quinton_hoover', 'forest_john_avon']:

    filename = Name + '.jpg'
    
    while True:
        print('Press "p" to take a picture of ' + filename)
        
        
    
        if PiOrUSB == 1: # PiCamerac
            rawCapture.truncate(0)
            # Press 'p' to take a picture
            for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    
                image = frame.array
                cv2.imshow("Card",image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("p"):
                    break
    
                rawCapture.truncate(0)
    
        if PiOrUSB == 2: # USB camera
            # Press 'p' to take a picture
            while(True):
    
                ret, frame = cap.read()
                try:
                    cv2.imshow("Card",frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("p"):
                        image = frame
                        break
                except Exception as e:
                    print(e)
                

        try:
            # Pre-process image
            thresh = Cards.preprocess_image(image)
            
            # try this for white border card
            thresh2 = Cards.preprocess_image(cv2.bitwise_not(image))
            if debug_pics: cv2.imwrite(debug_path + "0_prepare.jpg",thresh2)
            #
            
            # Find contours and sort them by size
            dummy,cnts,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts: 
                print(len(c))
            cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
#            cnts = [x for x in cnts if cv2.contourArea(x) > 50000]
        
            # Assume largest contour is the card. If there are no contours, print an error
            flag = 0
            image2 = image.copy()
        
            if len(cnts) == 0:
                print('No contours found!')
                quit()
        
            card = cnts[0]
        
            # Approximate the corner points of the card
            peri = cv2.arcLength(card,True)
            approx = cv2.approxPolyDP(card,0.01*peri,True)
            pts = np.float32(approx)
        
            x,y,w,h = cv2.boundingRect(card)
        
            # Flatten the card and convert it to 200x300
            warp = Cards.flattener(image,pts,w,h)
            if debug_pics: cv2.imwrite(debug_path + "1_rectangle.jpg",warp)
#            
#            # Grab corner of card image, zoom, and threshold
#            corner = warp[60:570, 0:630]
#            
#            if debug_pics: cv2.imwrite(debug_path + "2_corner.jpg",corner)
#        
#            corner_zoom = cv2.resize(corner, (0,0), fx=4, fy=4)
#            corner_blur = cv2.GaussianBlur(corner_zoom,(5,5),0)
#            
#            if debug_pics: cv2.imwrite(debug_path + "3_after_blur.jpg",corner_blur)
        
            dummy, cnts, hier = cv2.findContours(warp, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
        
            x,y,w,h = cv2.boundingRect(cnts[0])
        
            roi = warp[y:y+h, x:x+w]
            sized = cv2.resize(roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
            final_img = sized
        
            cv2.imshow("Image",final_img)
        
            # Save image
            print('Press "c" to save or "n" to proceed to next image.')
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                cv2.imwrite(img_path+filename,final_img)
                break
            elif key == ord('n'):
                break
            i = i + 1
        except Exception as e:
            print(e)

cv2.destroyAllWindows()

camera.close()
