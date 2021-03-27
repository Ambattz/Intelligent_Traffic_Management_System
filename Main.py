import cv2 as cv
import numpy as np
import time

def itms():
    def density():    
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

        def drawPred1(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
            cv.rectangle(cap1, (left, top), (right, bottom), (0, 0, 255))
            label = '%.2f' % conf
        # Get the label for the class name and its confidence
            if classes:
                assert (classId < len(classes))
                label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.putText(cap1, label, (left, top),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # Remove the bounding boxes with low confidence using non-maxima suppression


        def postprocess1(cap1, outs1):
            frameHeight = cap1.shape[0]
            frameWidth = cap1.shape[1]
            global count1
            count1 = 0
            classIds = []
            confidences = []
            boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs1:
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
                drawPred1(classIds[i], confidences[i], left,top, left + width, top + height)
                if classIds[i] in [1, 2, 3, 5, 7]:  # 0
                    count1 = count1 + 1


        def drawPred2(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
            cv.rectangle(cap1, (left, top), (right, bottom), (0, 0, 255))
            label = '%.2f' % conf
        # Get the label for the class name and its confidence
            if classes:
                assert (classId < len(classes))
                label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.putText(cap2, label, (left, top),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # Remove the bounding boxes with low confidence using non-maxima suppression


        def postprocess2(cap2, outs2):
            frameHeight = cap2.shape[0]
            frameWidth = cap2.shape[1]
            global count2
            count2 = 0
            classIds = []
            confidences = []
            boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs2:
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
                drawPred2(classIds[i], confidences[i], left,top, left + width, top + height)
                if classIds[i] in [1, 2, 3, 5, 7]:  # 0
                    count2 = count2 + 1



        def drawPred3(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
            cv.rectangle(cap3, (left, top), (right, bottom), (0, 0, 255))
            label = '%.2f' % conf
        # Get the label for the class name and its confidence
            if classes:
                assert (classId < len(classes))
                label = '%s:%s' % (classes[classId], label)
        # Display the label at the top of the bounding box
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.putText(cap3, label, (left, top),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    # Remove the bounding boxes with low confidence using non-maxima suppression


        def postprocess3(cap3, outs3):
            frameHeight = cap3.shape[0]
            frameWidth = cap3.shape[1]
            global count3
            count3 = 0
            classIds = []
            confidences = []
            boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs3:
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
                drawPred3(classIds[i], confidences[i], left,top, left + width, top + height)
                if classIds[i] in [1, 2, 3, 5, 7]:  # 0
                    count3 = count3 + 1


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
        outputFile = "output\ result1.png"
        outputFile1 = "output\ result2.png"
        outputFile2 = "output\ result3.png"
        outputFile3 = "output\ result4.png"
        


    # Process inputs
    # winName = 'DL OD with OpenCV'
    # cv.namedWindow(winName, cv.WINDOW_NORMAL)
    # cv.resizeWindow(winName, 1000, 1000)
    # if using an image, put the path of the image over here.
        cap = cv.imread("input\lane1.png")
        cv.imwrite(outputFile,cap)
        cap1 = cv.imread("input\lane2.png")
        cv.imwrite(outputFile1,cap1)
        cap2 = cv.imread("input\lane3.png")
        cv.imwrite(outputFile2,cap2)
        cap3 = cv.imread("input\lane4.png")
        cv.imwrite(outputFile3,cap3)

        

    # get frame from video
        # Create a 4D blob from a frame
        blob = cv.dnn.blobFromImage(cap, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Set the input to the net
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(cap, outs)


    # get frame from video
        # Create a 4D blob from a frame
        blob1= cv.dnn.blobFromImage(cap1, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Set the input to the net
        net.setInput(blob1)
        outs1 = net.forward(getOutputsNames(net))
        postprocess1(cap1, outs1)

    # get frame from video
        # Create a 4D blob from a frame
        blob2= cv.dnn.blobFromImage(cap2, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Set the input to the net
        net.setInput(blob2)
        outs2 = net.forward(getOutputsNames(net))
        postprocess2(cap2, outs2)

    # get frame from video
        # Create a 4D blob from a frame
        blob3= cv.dnn.blobFromImage(cap3, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        # Set the input to the net
        net.setInput(blob3)
        outs3 = net.forward(getOutputsNames(net))
        postprocess3(cap3, outs3)

    

#setting timer for each lane
    def timing(x):
        t=0
        if x>30:
            t=50
        elif (x>20 and x<30):
            t=40
        elif (x<20 and x>10):
            t=30
        else:
            t=20
        return t

    
    g = 10
    density()
    print("Vehicle Count lane no 1 " + str(count))
    print("Green signal ON for {} seconds".format(g))
    time.sleep(10)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density()
    print("Vehicle Count lane no 2 " + str(count1))
    print("Green signal ON for {} seconds".format(g))
    time.sleep(10)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density()
    print("Vehicle Count lane no 3 " + str(count2))
    print("Green signal ON for {} seconds".format(g))
    time.sleep(10)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")
    density()
    print("Vehicle Count Lane no 4 " + str(count3))
    print("Green signal ON for {} seconds".format(g))
    time.sleep(10)
    print("Yellow signal ON for 2 seconds")
    time.sleep(2)
    print("Red signal ON")
    print("")

    itms()

itms()
    # store the output image
    # cv.imwrite(outputFile, frame.astype(np.uint8))
    # show the output image
    # cv.imshow(winName, frame)
    # store the output video
    # Sample logic
    # if count > 20:
    #   Set timer to max value
    # else if count > 10:
    #   Set timer to lower value
    # else:
    #   Set timer to least value
