import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours

# [EDIT] edit here if change to other dir or run from other computers
imagepath="D:\\A-DEGREE\\Final Year Project\\DEVELOPMENT\\uitmOMRdetect\\StudentAnswers\\"
savebubbleregion = "D:\\A-DEGREE\\Final Year Project\\DEVELOPMENT\\uitmOMRdetect\\StudentAnswers\\"
image = cv2.imread(imagepath+"new2.jpeg")
im2 = image.copy()
kernel = np.ones((3, 3), np.uint8)


# [NOTICE] nothing here only resize for viewing purpose
print("image shape height, width, channel", image.shape)
r = 360 / image.shape[1]
dim = (360, int(image.shape[0] * r ))
image2 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# done resize for view


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
# [IMAGE PROCESSING] morphology > convert color > doGaussianBlur > doAdaptiveThreshold > do morphology closing

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

# [FEEDBACK] this doesnt work .. only sort contours working fine
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
# [SORT] sort contours according to the contourArea
    cnts = sorted(cnts, key=cv2.contourArea,reverse=True)
 
    # loop over the sorted contours
    for c in cnts:
    # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# [GET] the largest contours as Answer Sheet and apply approxDPblablabla
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

# [GET] bubble region .. its half of the page .. width dont have to divide by 2 
height,width,channel = paper.shape
cropheight = int(height/2)
bubbleregion = paper[cropheight:height , 0:width]

# [SAVE] save bubble region into the dir
# [NOTICE] edit here and adjust the saved name so its not redundant.. apply database here ?? or something timestamp rename ??
cv2.imwrite(savebubbleregion+"bubbleregion9.jpeg",bubbleregion)

brimage = bubbleregion.copy()

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
		print("done1")
	if i == 3:
		WidthStart = WidthStart - 15
		WidthEnd   = WidthEnd 
		print("done4")
	
	sect.insert(i,(bubbleregion[0:Y , WidthStart+35:WidthEnd]))
	WidthStart = WidthEnd
cv2.imshow("1",sect[0])
cv2.imshow("2",sect[1])
cv2.imshow("3",sect[2])
cv2.imshow("4",sect[3])


print("length sect ",len(sect))
cv2.waitKey(0)

# [NOTICE] begin process on bubbleregion
kernel = np.ones((3, 3), np.uint8)
#edit sini sakni kul 3:55
bubbleregion = doMorphologyEx(bubbleregion, cv2.MORPH_OPEN, kernel)
brcrop = cv2.cvtColor(bubbleregion,cv2.COLOR_BGR2GRAY)
brblur =  cv2.medianBlur(brcrop,9) 
#brblur = doGaussianBlur(brcrop,(9,9)) #xcomey
#brblur = doBlur(brcrop,(5,5)) #xleh guna sbb xdetect semo
#test	ret, thresh1 = cv2.threshold(blur1, 210, 255, cv2.THRESH_BINARY)
#thresh1 = cv2.threshold(blur1, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
brthresh = doAdaptiveThreshold(brblur)
kernel = np.ones((1, 1), np.uint8)
bthresh = doMorphologyEx(brthresh, cv2.MORPH_CLOSE, kernel)
#brthresh = doThreshold(brblur)[1]



# [FIND] find contours here after process
_,brcnts,_ = cv2.findContours(brthresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# done find contours

# [FILTER] filter the answer and bubble from others
newbrcnts = []
for c in brcnts:
	area = cv2.contourArea(c)
	if area > 75 and area < 250:
		newbrcnts.append(c)
		
# done process
questionCnts = []
 
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
		questionCnts.append(c)

questionCnts = contours.sort_contours(questionCnts,method="left-to-right")[0]

print(questionCnts[0].shape)


print("paper properties height, width, channel", paper.shape)
cv2.drawContours(bubbleregion,questionCnts, -1, (0, 255, 0), 1)
cv2.drawContours(brimage,newbrcnts, -1, (0, 255, 0), 1)
cv2.imshow("ni contour ", brimage)
cv2.imshow("ni question ", bubbleregion)
cv2.waitKey(0)
