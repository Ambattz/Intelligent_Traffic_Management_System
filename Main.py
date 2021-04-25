import cv2 as cv
import numpy as np
import time
from picamera import PiCamera

def density(cap):    
# Get the names of the output layers
    def getOutputsNames(net):
# Get the names of all the layers in the network
        layersNames = net.getLayerNames()
# Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Draw the predicted bounding box


    def drawPred(classId, conf, left, top, right, bottom):
# Draw a bounding box.
        cv.rectangle(cap, (left, top), (right, bottom), (0, 0, 255))
        label = '%.2f' % conf
# Get the label for the class name and its confidence
        if classes:
            assert (classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
# Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(
            label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.putText(cap, label, (left, top),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
# Remove the bounding boxes with low confidence using non-maxima suppression


    def postprocess(cap, outs):
        frameHeight = cap.shape[0]
        frameWidth = cap.shape[1]
        global count 
        count = 0
        classIds = []
        confidences = []
        boxes = []
# Scan through all the bounding boxes output from the network and keep only the
# ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left,top, left + width, top + height)
            if classIds[i] in [1, 2, 3, 5, 7]:  # 0
                count = count + 1



# Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image
# Load names of classes
    classesFile = "dependent-files\coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "dependent-files\yolov3.cfg"
    modelWeights = "dependent-files\yolov3.weights"
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    # if using image then make the output file a jpeg file.
     
    blob = cv.dnn.blobFromImage(cap, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # Set the input to the net
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(cap, outs)    


    return count



        # get frame from video
        # Create a 4D blob from a frame
    


outputFile1 = "output/result1.png"   
outputFile2 = "output/result2.png"   
outputFile3 = "output/result3.png"   
outputFile4 = "output/result4.png"   

    

#cap1 = cv.imread("input\lane1.png")


cap2 = cv.imread("input\lane2.png")

cap3 = cv.imread("input\lane3.png")

cap4 = cv.imread("input\lane4.png")

camera = PiCamera()
camera.start_preview()
sleep(5)
cap1 = camera.capture("input/lane1.png")
camera.stop_preview()

c1 = density(cap1)
print("Vehicle Count lane no 1 " + str(c1))
t1 = c1 + 1
print("Green signal ON for {} seconds".format(t1))
time.sleep(t1)
print("Yellow signal ON for 2 seconds")
time.sleep(2)
print("Red signal ON")
cv.imwrite(outputFile1,cap1)
print("")

c2 = density(cap2)
print("Vehicle Count lane no 2 " + str(c2))
t2 = c2 + 1
print("Green signal ON for {} seconds".format(t2))
time.sleep(t2)
print("Yellow signal ON for 2 seconds")
time.sleep(2)
print("Red signal ON")
cv.imwrite(outputFile2,cap2)
print("")

c3 = density(cap3)
print("Vehicle Count lane no 3 " + str(c3))
t3 = c3 + 1
print("Green signal ON for {} seconds".format(t3))
time.sleep(t3)
print("Yellow signal ON for 2 seconds")
time.sleep(2)
print("Red signal ON")
cv.imwrite(outputFile3,cap3)
print("")

c4 = density(cap4)
print("Vehicle Count Lane no 4 " + str(c4))
t4 = c4 + 1
print("Green signal ON for {} seconds".format(t4))
time.sleep(t4)
print("Yellow signal ON for 2 seconds")
time.sleep(2)
print("Red signal ON")
cv.imwrite(outputFile4,cap4)
print("")

   
