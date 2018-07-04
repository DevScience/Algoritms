import cv2
import sys

#read the image
image = cv2.imread(sys.argv[1])
cv2.imshow("Image", image)
cv2.waitKey(0)

python display.py ship.jpg