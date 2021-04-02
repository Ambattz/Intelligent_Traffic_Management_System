# import the necessary packages
import numpy as np
#import argparse
import cv2
import time

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True, help='path to the input image')
# # ap.add_argument('-p', '--prototxt', default='/Users/siddhantbansal/Desktop/Python/Personal_Projects/Object_Detection/MobileNetSSD_deploy.prototxt.txt', help='path to Caffe deploy prototxt file')
# # ap.add_argument('-m', '--model', default='/Users/siddhantbansal/Desktop/Python/Personal_Projects/Object_Detection/MobileNetSSD_deploy.caffemodel', help='path to the Caffe pre-trained model')
# ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe deploy prototxt file')
# ap.add_argument('-m', '--model', required=True, help='path to the Caffe pre-trained model')
# ap.add_argument('-c', '--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
# args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

def itms():
    def density(image):
        count = 0
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the neural network
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., the probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
                # extract the index of the classes label from the 'detections',
                # then compute the (x, y)-coordinates of the bounding box for the object
            if confidence > 0.000000000000001:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                    # display the prediction
                label = '{}: {:.2f}%'.format(CLASSES[idx], confidence * 100)
                count = count + 1
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        return count

    def timing(x):
            t=0
            if x>30:
                t=60
            elif (x>20 and x<30):
                t=40
            elif (x<20 and x>10):
                t=30
            else:
                t=20
            return t


    outputFile = "output\ result1.png"
    outputFile1 = "output\ result2.png"
    outputFile2 = "output\ result3.png"
    outputFile3 = "output\ result4.png"

    cap1 = cv2.imread("input/lane1.png")
    c1 = density(cap1)
    cv2.imwrite(outputFile,cap1)


    cap2 = cv2.imread("input/lane2.png")
    c2 = density(cap2)
    cv2.imwrite(outputFile,cap2)

    cap3 = cv2.imread("input/lane3.png")
    c3 = density(cap3)
    cv2.imwrite(outputFile,cap3)

    cap4 = cv2.imread("input/lane4.png")
    c4 = density(cap4)
    cv2.imwrite(outputFile,cap4)
    

    print("Vehicle Count lane no 1 " + str(c1))
    t1 = timing(c1)
    print("Green signal ON for {} seconds".format(t1))
    time.sleep(t1)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density(cap2)
    density(cap3)
    density(cap4)
    print("Vehicle Count lane no 2 " + str(c2))
    t2 = timing(c2)
    print("Green signal ON for {} seconds".format(t2))
    time.sleep(t2)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density(cap3)
    density(cap4)
    print("Vehicle Count lane no 3 " + str(c3))
    t3 = timing(c3)
    print("Green signal ON for {} seconds".format(t3))
    time.sleep(t3)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density(cap4)
    print("Vehicle Count Lane no 4 " + str(c4))
    t4 = timing(c4)
    print("Green signal ON for {} seconds".format(t4))
    time.sleep(t4)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")

    itms()

itms()