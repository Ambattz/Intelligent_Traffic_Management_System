import cv2
import numpy
# capture frames from a video
path1 = r'D:\Study\Project\ITMS\input\lane1.mp4'
cap1 = cv2.VideoCapture(path1)
path2 = r'D:\Study\Project\ITMS\input\lane2.mp4'
cap2 = cv2.VideoCapture(path2)
path3 = r'D:\Study\Project\ITMS\input\lane3.mp4'
cap3 = cv2.VideoCapture(path3)
path4 = r'D:\Study\Project\ITMS\input\lane4.mp4'
cap4 = cv2.VideoCapture(path4)
# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')
# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames1 = cap1.read()
    ret, frames2 = cap2.read()
    ret, frames3 = cap3.read()
    ret, frames4 = cap4.read()
    # convert to gray scale of each frames
    gray1 = cv2.cvtColor(frames1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frames2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frames3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(frames4, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars1 = car_cascade.detectMultiScale(gray1, 1.1, 1)
    cars2 = car_cascade.detectMultiScale(gray2, 1.1, 1)
    cars3 = car_cascade.detectMultiScale(gray3, 1.1, 1)
    cars4 = car_cascade.detectMultiScale(gray4, 1.1, 1)
    # To draw a rectangle in each cars
    for (x, y, w, h) in cars1:
        cv2.rectangle(frames1, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in cars2:
        cv2.rectangle(frames2, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in cars3:
        cv2.rectangle(frames3, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in cars4:
        cv2.rectangle(frames4, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Display frames in a window
    frames1 = numpy.concatenate((frames1, frames2), axis=1)
    frames2 = numpy.concatenate((frames3, frames4), axis=1)
    frames = numpy.concatenate((frames1, frames2), axis=0)
    cv2.imshow('LANE-CAPTURING', frames)
    # cv2.imshow('LANE-02', frames2)
    # cv2.imshow('LANE-03', frames3)
    # cv2.imshow('LANE-04', frames4)
    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break
# De-allocate any associated memory usage
cv2.destroyAllWindows()
