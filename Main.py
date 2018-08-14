#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  
#  Copyright 2018 Syirasky <Syirasky@DESKTOP-17Q7VC8>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import math
import imutils
import time
from pandas import DataFrame


def doMorphologyEx(im,method,kern):
	out = cv2.morphologyEx(im, method, kernel)
	return out
	
def doAdaptiveThreshold(image):
	out = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	return out

def doGaussianBlur(im,numhere):
	out = cv2.GaussianBlur(im,numhere ,0)
	return out
	
def doMedianBlur(im,numhere):
	out = cv2.medianBlur(im,numhere)
	return out

def doBlur(im,numhere):
	out = cv2.blur(im,numhere)
	return out

def doThreshold(im):
	out = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
	return out

def get_contour_precedence(cnt, cols):
    tolerance_factor = 25
    origin = cv2.boundingRect(cnt)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
def findAllCnts(image):
	# [IMAGE PROCESSING] morphology opening > convert color > doGaussianBlur > doAdaptiveThreshold > do morphology closing

	image = doMorphologyEx(image, cv2.MORPH_OPEN, kernel)
	graycrop = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#blur1 =  cv2.medianBlur(graycrop,5)
	blur1 = doGaussianBlur(graycrop,(5,5))
	#test	ret, thresh1 = cv2.threshold(blur1, 210, 255, cv2.THRESH_BINARY)
	#thresh1 = cv2.threshold(blur1, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh1 = doAdaptiveThreshold(blur1)

	thresh1 = doMorphologyEx(thresh1,cv2.MORPH_CLOSE,kernel)
	# done image process

	# [FIND] find contours here after process
	_,cnts,_ = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	# done find contours
	
	
	docCnt = None
	print(len(cnts))
	# ensure that at least one contour was found
	return cnts
	
"""

# [FEEDBACK] this doesnt work .. only sort contours working fine
 code here // find at experimentcode
"""

	

def findpaper(cnts):# [GET] the largest contours as Answer Sheet and apply approxDPblablabla
	cnts = sorted(cnts, key=cv2.contourArea,reverse=True)

	for i in range(2):
		area = cv2.contourArea(cnts[i])
		print(i,"<",area)
	 
	peri = cv2.arcLength(cnts[0], True)
	approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
	docCnt = approx
	# done find paper

	# apply a four point perspective transform to both the
	# original image and grayscale image to obtain a top-down
	# birds eye view of the paper
	paper = four_point_transform(im2, docCnt.reshape(4, 2))
	return paper


def findIdRegion(cnts):
	height,width,channel = paper.shape
	startheight = int(height/4)
	endheight = int(height/2)
	startwidth = int(width/2)
	endwidth = width
	
	idRegion = paper[startheight:endheight, startwidth+50:endwidth]
	
	return idRegion
	
	
def getbubbleregion(paper,nametosave):
	# [GET] bubble region .. its half of the page .. width dont have to divide by 2 
	height,width,channel = paper.shape
	cropheight = int(height/2)
	bubbleregion = paper[cropheight:height , 0:width]

	# [SAVE] save bubble region into the dir
	# [NOTICE] edit here and adjust the saved name so its not redundant.. apply database here ?? or something timestamp rename ??
	cv2.imwrite(savebubbleregion+nametosave+".jpeg",bubbleregion)

	brimage = bubbleregion.copy()
	return bubbleregion
	
	
def divideSection(bubbleregion):

	# [GET] region of every questions section
	height,width = bubbleregion.shape[:2]
	print("height", height)
	print("width", width)

	Y = height
	WidthStart = 0
	sect = list()
	for i in range(4):
		WidthEnd = int(width * (i+1)/4)
		print(i)
		if i == 0:
			WidthStart = WidthStart + 5
			
		if i == 3:
			WidthStart = WidthStart - 15
			WidthEnd   = WidthEnd 
			
		
		sect.insert(i,(bubbleregion[0:Y , WidthStart+35:WidthEnd]))
		WidthStart = WidthEnd
	return sect



def findBubble(img):
	# [FIND] find contours here after process
	#edit sini sakni kul 3:55 kat bubbleregion ganti sect[i]
	kernel = np.ones((3, 3), np.uint8)
	bubbleregion = doMorphologyEx(img, cv2.MORPH_OPEN, kernel)
	brcrop = cv2.cvtColor(bubbleregion,cv2.COLOR_BGR2GRAY)
	brblur =  cv2.medianBlur(brcrop,9)
	brthresh = doAdaptiveThreshold(brblur)
	kernel = np.ones((3, 3), np.uint8)
	bthresh = doMorphologyEx(brthresh, cv2.MORPH_CLOSE, kernel)
	_,brcnts,_ = cv2.findContours(brthresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	# done find contours

	# [FILTER] filter the bubble from other contours
	print("brcnts length ",len(brcnts))

	newbrcnts = []
	for c in brcnts:
		area = cv2.contourArea(c)
		perimeter = cv2.arcLength(c,True)
		
		if area > 50 and area < 250:
			newbrcnts.append(c)
	# done process
	bubblecnts = []
	 
	# loop over the contours
	for c in newbrcnts:
		# compute the bounding box of the contour, then use the
		# bounding box to derive the aspect ratio
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
	 
		# in order to label the contour as a question, region
		# should be sufficiently wide, sufficiently tall, and
		# have an aspect ratio approximately equal to 1
		if w >= 7 and h >= 7 and ar >= 0.1 and ar <= 1.2:
			bubblecnts.append(c)

	# [SORT] sort contours 1	

	# [SORT] sort contours 2

	bubblecnts.sort(key=lambda x:get_contour_precedence(x, sect[0].shape[1]))    
	# [DONE] done contours extract and sort
		
	
	return bubblecnts


def findNonZeroValue(img,cnt):  
	row = 1
	k = 0
	bulat = []
	for i in range(len(cnt)):
		if k > 4:
			k = 0
			row = 1 + row 
		k = k + 1
		
		markedgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ansblur = doGaussianBlur(markedgray,(3,3))
		ansthresh = doAdaptiveThreshold(ansblur)
		ansthresh =  cv2.threshold(ansthresh, 0, 255, cv2.THRESH_BINARY_INV)[1]
		bubbled = None
		mask = np.zeros(ansthresh.shape, dtype="uint8")
		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(ansthresh, ansthresh, mask=mask)
		cv2.drawContours(mask,cnt,i, 255, -1)
		total = cv2.countNonZero(mask)
		bulat.append(total)
		mask = np.zeros(ansthresh.shape,dtype="uint8")
		print(row,",",i,": bulat <", bulat[i])
	return bulat

 
def getBlackBubble(AnsListHere,bubbleNum):
	ansnum = []
	for l,rowfirstnum in enumerate(np.arange(0,len(studentanswer),5)):
		min = 1000
		i = rowfirstnum #start with next row number
		
		for c in range(5):
			if i < bubbleNum:
				lorek = studentanswer[i] #bubble length is 75 , thats why put i it start with next row first num , exp ; 0 , 5 , 10
				# code utk check min lebih kurey (- +) dgn lorek ko
				if lorek < min:
					min = lorek
					selected = c
				
				i = i + 1
			else:
				i = 0
		ansnum.append(selected)
	return ansnum


def findIdContour(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel =  np.ones((5, 5), np.uint8)
	gray =  doMorphologyEx(gray, cv2.MORPH_OPEN, kernel)
	#blurred = cv2.GaussianBlur(gray, (1, 1), 0)
	blurred = cv2.medianBlur(gray,3)
	#th2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,115,2)
	th2 = cv2.Canny(blurred, 160, 200)
	th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
	_,cnts,_ = cv2.findContours(th2, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	filtered_cnts = []
	for i in cnts:
		area = cv2.contourArea(i)
		if area > 600 and area < 2000:
			filtered_cnts.append(i)
	return cnts
	
def resizeimage(image):
	print("image shape height, width, channel", image.shape)
	r = 360 / image.shape[1]
	dim = (360, int(image.shape[0] * r ))
	image2 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return image2
	
def getTotalMark(studentanswer,correctanswer,bil):
	score = 0

	for i in range(bil):
		if studentanswer[i]==correctanswer[i]:
			score = 1 + score
	return score
	
"""
   ___|  |                |        _ \                                          \                                      
 \___ \  __|   _` |   __| __|     |   |  __| _ \   __|   _ \   __|   __|       _ \    __ \    __| \ \  \   / _ \   __| 
       | |    (   |  |    |       ___/  |   (   | (      __/ \__ \ \__ \      ___ \   |   | \__ \  \ \  \ /  __/  |    
 _____/ \__| \__,_| _|   \__|    _|    _|  \___/ \___| \___| ____/ ____/    _/    _\ _|  _| ____/   \_/\_/ \___| _|    
                                                                                                                       
"""
StudentId= "2017123456"
ABCDvalue = {0:'A',1:'B',2:'C',3:'D',4:'E',99:'Error'}
CorrectAnswer = [1,0,3,2,2,3,0,2,2,0,2,1,1,0,2,0,3,1,3,1] #modify this so examiner can submit the answer

# [EDIT] edit here if change to other dir or run from other computers
imagepath="D:\\A-DEGREE\\Final Year Project\\DEVELOPMENT\\OMR-MCQ-Detection\\StudentAnswers\\"
savebubbleregion = "D:\\A-DEGREE\\Final Year Project\\DEVELOPMENT\\OMR-MCQ-Detection\\StudentAnswers\\BubbleRegion\\"
saveidregion = "D:\\A-DEGREE\Final Year Project\\DEVELOPMENT\\OMR-MCQ-Detection\\StudentAnswers\\StudentIdRegion\\"
image = cv2.imread(imagepath+"newFullPage.jpg")
bil = 20 #mod this because question numbers can be vary
im2 = image.copy()
kernel = np.ones((3, 3), np.uint8)


# [NOTICE] nothing here only resize for viewing purpose
image2 = resizeimage(image)
# done resize for view

# [NOTICE] find paper
cnts = findAllCnts(image)
paper = findpaper(cnts)

# [NOTICE] begin process on bubbleregion

bubbleregion = getbubbleregion(paper,"test")
sect = divideSection(bubbleregion)
partans = []
ansnum = []
thecnt = findBubble(sect[0])
cv2.drawContours(sect[0],thecnt,-1,(0,0,255),0)
cv2.imshow("huha",sect[0])
for i in range(2):
	thecnt = findBubble(sect[i])
	studentanswer = findNonZeroValue(sect[i],thecnt)
	partans = getBlackBubble(studentanswer,len(studentanswer))
	ansnum.extend(partans)
	

print("\nStudent Answers'")
if len(ansnum) > 0:
	score = getTotalMark(ansnum,CorrectAnswer,bil)
	print("Score",score)
	for i in range(bil):
		print(i+1,ABCDvalue[ansnum[i]])

	

	
	
	
cv2.waitKey(0)
