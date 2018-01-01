# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:45:50 2017

@author: rounayak
"""

import argparse
import cv2
from PIL import Image
import os 
import pandas as pd
from tqdm import tqdm
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
        
        
        
      
        
        
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
m=str(args.values()).strip('[').strip(']').strip("'")
print(m)
size=1920,1357
from PIL import Image
#im = Image.open('1.tif')
im = Image.open(m)
im.save('dum.png')
im = Image.open("dum.png")
im_resized = im.resize(size, Image.ANTIALIAS)
im_resized.save("dum.png", "PNG") 
# load the image, clone it, and setup the mouse callback function
#image = cv2.imread(args["image"])
image = cv2.imread('dum.png')
clone = image.copy()
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# if there are two reference points, then crop the region of interest
# from the image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        print(refPt[0][1],refPt[0][0],refPt[1][1],refPt[1][0])
        
         
   
	cv2.imshow("ROI", roi)
   
	cv2.waitKey(0)
 
# close all open windows
cv2.destroyAllWindows()    










def looper(f,df): 
 import pandas as pd
    
 size=1920,1357
 from PIL import Image
 im = Image.open(f)
 im.resize(size, Image.ANTIALIAS).save('dum.png',"PNG")

 import os
 import cv2
 img = cv2.imread("dum.png")
 crop_img = img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
  
 cv2.imwrite("cropped.png", crop_img)
 import pytesseract
 from PIL import Image

 pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'
 tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
 y=Image.open('cropped.png')
 output= pytesseract.image_to_string(y)
 output1=output.encode('utf-8')
 #print output
 #file = open("output1.txt","w")
 #file.write(output)
 #file.close()
 
 
 x1=len(df.index)
 #print(x1)
 
 
 df.loc[x1,'File']=f
 df.loc[x1,'Value']=output1
 return(df)
# df.loc[x1,'Value']=output.split[1]
 #df.to_csv("Extraction.csv",index=False)
ext='f' 
dir_path = os.getcwd()
from PIL import Image
df=pd.DataFrame({'File':[],'Key':[],'Value':[]})
df.to_csv("Extraction.csv", index= False)
files = os.listdir(dir_path)
files=[i  for i in files if i.endswith(ext)]
[looper(x,df) for x in tqdm(files)]   
df.to_csv("Extraction.csv",index=False)    

df=pd.read_csv("Extraction.csv")
x=len(df.index)  


df['Key']='Date'
    
df.to_csv("Extraction.csv",index=False)   


      
df=pd.read_csv("Extraction.csv")
x=len(df.index) 
for i in range(x):
    
    df.loc[i,'Value']='.'.join([x.strip() for x in str(df.loc[i,'Value']).replace(",",".").split('.')])
    
df.to_csv("Extraction2.csv",index=False)  
os.remove('Extraction.csv') 
os.remove('dum.png') 
os.remove('cropped.png') 
 

    
 
     
       