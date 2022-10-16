from sre_constants import SUCCESS
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap= cv2.VideoCapture (0)
detector = HandDetector(maxHands=2)


offset=20 #phần bo
imgSize = 300 #thiết lập size ảnh

folder="D:\Workspace\{}ghiencuu\hand_machine_learning\Data_image\C_image".format('n')
stemp=0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        if len(hands)==1:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 #ones:setup matrix , has value: 8bit(0-255) => uint8 : số nguyên không dấu của 8bit
            #0-255 là màu đen => *255 để ra màu trắng
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]


            imgCropShape = imgCrop.shape #matrix có h,w,z

             #đưa Crop vào white để giữ nguyên khung hình
            
            
            aspectRatio = h/w #tỉ lệ khung hình
            
            #để giữ nguyên 1 chiều cao và thay đổi chiều rộng theo chiều cao
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
            
            cv2.imshow('ImageCrop',imgCrop)
            cv2.imshow('ImageWhite',imgWhite)
        # if len(hands)==2:
        #     hand = hands[0]
        #     hand2 =hands[1]
        #     x,y,w,h = hand['bbox']
        #     x2,y2,w2,h2 = hand2['bbox']
        #     imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        #     imgCrop2 = img[y2-offset:y2+h2+offset,x2-offset:x2+w2+offset]
        #     cv2.imshow('ImageCrop',imgCrop)
        #     cv2.imshow('ImageCrop2',imgCrop2)
        
    cv2.imshow('Image',img)
    key=cv2.waitKey(1)
    if key == ord("s"):
        stemp +=1
        cv2.imwrite(f'{folder}/Image_{stemp}_{time.time()}.jpg',imgWhite)
        print(stemp)
        if stemp == 300:
            break
