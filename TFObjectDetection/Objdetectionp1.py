import cv2 # open cv
import numpy as np 
from matplotlib import pyplot as plt

#image read, and type of filter applied
img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
#show
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
using plt 
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()
'''
cv2.imwrite('watchgray.png',img)
