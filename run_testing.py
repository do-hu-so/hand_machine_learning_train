from cProfile import label
from sre_constants import SUCCESS
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier #phân loại module
import numpy as np
import math
import time
import tensorflow

cap= cv2.VideoCapture (0)
detector = HandDetector(maxHands=2) #trình phát hiện
classsifier = Classifier("D:\Workspace\{}ghiencuu\hand_machine_learning\Model\keras_model.h5".format('n'),"D:\Workspace\{}ghiencuu\hand_machine_learning\Model\labels.txt".format('n')) #phân loại (module,nhãn)

offset=20 #phần bo
imgSize = 300 #thiết lập size ảnh

stemp=0

labels=["A","B","C"]

def hand_prediction(imgSize,x,y,w,h):
            imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 #ones:setup matrix , has value: 8bit(0-255) => uint8 : số nguyên không dấu của 8bit
            #0-255 là màu đen => *255 để ra màu trắng
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCropShape = imgCrop.shape #matrix có h,w,z                       
            aspectRatio = h/w
            if aspectRatio > 1 :
                #set khung cho h>w
                k= imgSize/h
                wCal= math.ceil(k*w) #luon lam tròn đến số cao hơn, vd 3.2 or 3.5 => 4
                imgResize = cv2.resize(imgCrop,(wCal,imgSize)) #thay doi size
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal)/2)               
                #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]]=imgResize #cũ
                #0 là điểm bắt đầu, ket thuc là chiều cao của phần cắt hình ảnh
                imgWhite[:,wGap:wCal+wGap]=imgResize          
            else:
                #set khung cho h<w
                k= imgSize/w
                hCal= math.ceil(k*h) #luon lam tròn đến số cao hơn, vd 3.2 or 3.5 => 4
                imgResize = cv2.resize(imgCrop,(imgSize,hCal)) #thay doi size
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal)/2)               
                imgWhite[hGap:hCal+hGap,:]=imgResize
                
            prediction,index =classsifier.getPrediction(imgWhite,draw=False)#dự đoán và chỉ số
            print(prediction,index)

            cv2.rectangle(imgOutput,(x-offset,y-offset-50),
                                    (x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED) #bo putText
            cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            cv2.rectangle(imgOutput,(x-offset,y-offset),
                                    (x+w+offset,y+h+offset),(255,0,255),4)
            #cv2.imshow('ImageCrop',imgCrop)
            cv2.imshow('ImageWhite',imgWhite)

                
            
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        if len(hands)==1:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            hand_prediction(imgSize,x,y,w,h)

            
        if len(hands)==2:
            hand = hands[0]
            hand2 =hands[1]
            x,y,w,h = hand['bbox']
            x2,y2,w2,h2 = hand2['bbox']
            hand_prediction(imgSize,x,y,w,h)
            hand_prediction(imgSize,x2,y2,w2,h2)
            
        
    cv2.imshow('Image',imgOutput)
    cv2.waitKey(1)


