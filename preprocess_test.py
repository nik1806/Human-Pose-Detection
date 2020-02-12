
"""
Reshaping the test images to (640, 360)
"""

import cv2
import glob

i = 0
name = 'test_cases/'

action = ['shoot','normal','kick','slap']

for act in action:
    for filename in glob.glob(name+act+'/*'):
        i+=1
        image = cv2.imread(filename)
        res_image = cv2.resize(image,(640,360))
        cv2.imwrite(name+act+str(i)+'.jpg',res_image)
 #   cv2.imwrite(filename,res_image)   