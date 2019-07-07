import cv2
import numpy as np
import argparse
import matplotlib

def nothing(*arg):
        pass

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help = "path to the input video")
ap.add_argument("-w", "--write", help = "write masked images to file", action="store_true")
ap.add_argument("-d", "--delay", help = "delay between images", default=500)
args = vars(ap.parse_args())

initialColor = (0, 100, 80, 10, 255, 255)

cap = cv2.VideoCapture(args["input"])

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

cv2.namedWindow('colorFilter')
# sliders for low HSV value
cv2.createTrackbar('lowHue', 'colorFilter', initialColor[0], 255, nothing)
cv2.createTrackbar('lowSat', 'colorFilter', initialColor[1], 255, nothing)
cv2.createTrackbar('lowVal', 'colorFilter', initialColor[2], 255, nothing)
# sliders for high HSV value
cv2.createTrackbar('highHue', 'colorFilter', initialColor[3], 255, nothing)
cv2.createTrackbar('highSat', 'colorFilter', initialColor[4], 255, nothing)
cv2.createTrackbar('highVal', 'colorFilter', initialColor[5], 255, nothing)

counter = 0
doWrite = args["write"]
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:

    lowHue = cv2.getTrackbarPos('lowHue', 'colorFilter')
    lowSat = cv2.getTrackbarPos('lowSat', 'colorFilter')
    lowVal = cv2.getTrackbarPos('lowVal', 'colorFilter')
    highHue = cv2.getTrackbarPos('highHue', 'colorFilter')
    highSat = cv2.getTrackbarPos('highSat', 'colorFilter')
    highVal = cv2.getTrackbarPos('highVal', 'colorFilter')

    # convert frame to HSV and mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorLow = np.array([lowHue, lowSat, lowVal], dtype="uint8")
    colorHigh = np.array([highHue, highSat, highVal], dtype="uint8")
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    result = cv2.bitwise_and(frame, frame, mask = mask)

    l = np.zeros([5,5,3])
    h = np.zeros([5,5,3])
    cLow = matplotlib.colors.hsv_to_rgb(colorLow/255.0)
    cHigh = matplotlib.colors.hsv_to_rgb(colorHigh/255.0)

    l[:,:,0] = cLow[2]
    l[:,:,1] = cLow[1]
    l[:,:,2] = cLow[0]

    h[:,:,0] = cHigh[2]
    h[:,:,1] = cHigh[1]
    h[:,:,2] = cHigh[0]

    red = np.zeros(frame.shape, frame.dtype)
    red[:,:] = (0, 0, 255)
    redMask = cv2.bitwise_and(red, red, mask=mask)

    frameMasked = cv2.addWeighted(redMask, 1, frame, 1, 0)

    #cv2.imshow('Frame', np.hstack([frame, result]))
    cv2.imshow('Frame', np.hstack([frameMasked, frame]))
    cv2.imshow('Colors', np.hstack([l, h]))

    counter = counter + 1

    if doWrite:
        cv2.imwrite(f"frame{counter}.png", np.hstack([frameMasked, frame]))
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(args["delay"]) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
