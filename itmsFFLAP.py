import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import six.moves.urllib as urllib
import sys
from goto import with_goto
import threading
from threading import Thread
from multiprocessing import Process,Lock
import tarfile
#import RPi.GPIO as GPIO
#GPIO.setwarnings(False)
import time
from prettytable import PrettyTable
from datetime import date
from datetime import datetime
import tensorflow.compat.v1 as tf
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# script repurposed from sentdex's edits and TensorFlow's example script. Pretty messy as not all unnecessary
# parts of the original have been removed
# # Model preparation
# ## Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
lock=threading.Lock()
lockg=threading.Lock()
'''maxlimit = threading.Semaphore(2)
#GPIO OUTPUT SET
GPIO.setmode(GPIO.BOARD)
GPIO.setup(36,GPIO.OUT)
GPIO.setup(38,GPIO.OUT)
GPIO.setup(40,GPIO.OUT)
GPIO.setup(8,GPIO.OUT)
GPIO.setup(10,GPIO.OUT)
GPIO.setup(12,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
GPIO.setup(15,GPIO.OUT)
GPIO.setup(19,GPIO.OUT)
GPIO.setup(21,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)
#ALLOFF
GPIO.output(38,0)
GPIO.output(10,0)
GPIO.output(13,0)
GPIO.output(21,0)
GPIO.output(12,0)
GPIO.output(15,0)
GPIO.output(23,0)
GPIO.output(40,0)
GPIO.output(36,0)
GPIO.output(8,0)
GPIO.output(11,0)
GPIO.output(19,0)
#RED-ON
GPIO.output(12,1)
GPIO.output(15,1)
GPIO.output(23,1)
GPIO.output(40,1)'''

MODEL_NAME = 'trained_model'  # change to whatever folder has the new graph
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'label.pbtxt')  # our labels are in training/object-detection.pbkt
NUM_CLASSES = 3  # we only are using one class at the moment (mask at the time of edit)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# In[7]:
def gpio(lock,emergency,t):
    lock.acquire()
    print("GPIO Thread Activated")
    time.sleep(40)
    print("")
    print("-----------------------------------------------------------------")
    '''#LANE01
    if(4==len(emergency)):
        if(emergency[1] == 1):
            print("Emergency Vehicle Detected in Lane 2")
            #gpio_lane2
            GPIO.output(12,0)
            GPIO.output(10,1)
            time.sleep(2)
            GPIO.output(10,0)
            GPIO.output(8,1) 
            time.sleep(t[1])
            GPIO.output(8,0)             
            GPIO.output(10,1)            
            time.sleep(2)                
            GPIO.output(10,0)
            GPIO.output(12,1)
    if(4==len(emergency)):
        if(emergency[2] == 1):
            print("Emergency Vehicle Detected in Lane 3")
            #gpio_lane3
            GPIO.output(15,0)
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0) 
            GPIO.output(11,1) 
            time.sleep(t[2])
            GPIO.output(11,0)             
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0)                       
            GPIO.output(15,1)
    if(4==len(emergency)):
        if(emergency[3] == 1):
            print("Emergency Vehicle Detected in Lane 4 ")
            #gpio_lane 4
            GPIO.output(23,0)
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0) 
            GPIO.output(19,1) 
            time.sleep(t[3])
            GPIO.output(19,0)             
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0)            
            GPIO.output(23,1)
    GPIO.output(40,0)
    GPIO.output(38,1)            
    time.sleep(2)                
    GPIO.output(38,0)
    GPIO.output(36,1) 
    time.sleep(t[0])
    GPIO.output(36,0)             
    GPIO.output(38,1)            
    time.sleep(2)                
    GPIO.output(38,0)
    GPIO.output(40,1)
    if(4==len(emergency)):
        if(emergency[1] == 1):
            print("Emergency Vehicle Detected in Lane 2")
            #gpio_lane2
            GPIO.output(12,0)
            GPIO.output(10,1)
            time.sleep(2)
            GPIO.output(10,0)
            GPIO.output(8,1) 
            time.sleep(t[1])
            GPIO.output(8,0)             
            GPIO.output(10,1)            
            time.sleep(2)                
            GPIO.output(10,0)
            GPIO.output(12,1)
    if(4==len(emergency)):
        if(emergency[2] == 1):
            print("Emergency Vehicle Detected in Lane 3")
            #gpio_lane3
            GPIO.output(15,0)
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0) 
            GPIO.output(11,1) 
            time.sleep(t[2])
            GPIO.output(11,0)             
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0)                       
            GPIO.output(15,1)
    if(4==len(emergency)):
        if(emergency[3] == 1):
            print("Emergency Vehicle Detected in Lane 4 ")
            #gpio_lane 4
            GPIO.output(23,0)
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0) 
            GPIO.output(19,1) 
            time.sleep(t[3])
            GPIO.output(19,0)             
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0)            
            GPIO.output(23,1)
    #LANE02
    GPIO.output(12,0)
    GPIO.output(10,1)
    time.sleep(2)
    GPIO.output(10,0)
    GPIO.output(8,1) 
    time.sleep(t[1])
    GPIO.output(8,0)             
    GPIO.output(10,1)            
    time.sleep(2)                
    GPIO.output(10,0)
    GPIO.output(12,1)
    if(4==len(emergency)):
        if(emergency[0] == 1):
            print("Emergency Vehicle Detected in Lane 1")    
            #gpio_lane1
            GPIO.output(40,0)
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(36,1) 
            time.sleep(t[0])
            GPIO.output(36,0)             
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(40,1)
    if(4==len(emergency)):
        if(emergency[2] == 1):
            print("Emergency Vehicle Detected in Lane 3")
            #gpio_lane3
            GPIO.output(15,0)
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0) 
            GPIO.output(11,1) 
            time.sleep(t[2])
            GPIO.output(11,0)             
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0)                       
            GPIO.output(15,1)
    if(4==len(emergency)):
        if(emergency[3] == 1):
            print("Emergency Vehicle Detected in Lane 4 ")
            #gpio_lane 4
            GPIO.output(23,0)
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0) 
            GPIO.output(19,1) 
            time.sleep(t[3])
            GPIO.output(19,0)             
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0)            
            GPIO.output(23,1)
    #LANE03
    GPIO.output(15,0)
    GPIO.output(13,1)            
    time.sleep(2)                
    GPIO.output(13,0) 
    GPIO.output(11,1) 
    time.sleep(t[2])
    GPIO.output(11,0)             
    GPIO.output(13,1)            
    time.sleep(2)                
    GPIO.output(13,0)                       
    GPIO.output(15,1)
    if(4==len(emergency)):
        if(emergency[0] == 1):
            print("Emergency Vehicle Detected in Lane 1")    
            #gpio_lane1
            GPIO.output(40,0)
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(36,1) 
            time.sleep(t[0])
            GPIO.output(36,0)             
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(40,1)
    if(4==len(emergency)):
        if(emergency[1] == 1):
            print("Emergency Vehicle Detected in Lane 2")
            #gpio_lane2
            GPIO.output(12,0)
            GPIO.output(10,1)
            time.sleep(2)
            GPIO.output(10,0)
            GPIO.output(8,1) 
            time.sleep(t[1])
            GPIO.output(8,0)             
            GPIO.output(10,1)            
            time.sleep(2)                
            GPIO.output(10,0)
            GPIO.output(12,1)
    if(4==len(emergency)):
        if(emergency[3] == 1):
            print("Emergency Vehicle Detected in Lane 4 ")
            #gpio_lane 4
            GPIO.output(23,0)
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0) 
            GPIO.output(19,1) 
            time.sleep(t[3])
            GPIO.output(19,0)             
            GPIO.output(21,1)            
            time.sleep(2)                
            GPIO.output(21,0)            
            GPIO.output(23,1)
    #LANE04
    GPIO.output(23,0)
    GPIO.output(21,1)            
    time.sleep(2)                
    GPIO.output(21,0) 
    GPIO.output(19,1) 
    time.sleep(t[3])
    GPIO.output(19,0)             
    GPIO.output(21,1)            
    time.sleep(2)                
    GPIO.output(21,0)            
    GPIO.output(23,1)           
    GPIO.output(23,1)
    if(4==len(emergency)):
        if(emergency[0] == 1):
            print("Emergency Vehicle Detected in Lane 1")    
            #gpio_lane1
            GPIO.output(40,0)
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(36,1) 
            time.sleep(t[0])
            GPIO.output(36,0)             
            GPIO.output(38,1)            
            time.sleep(2)                
            GPIO.output(38,0)
            GPIO.output(40,1)
    if(4==len(emergency)):
        if(emergency[1] == 1):
            print("Emergency Vehicle Detected in Lane 2")
            #gpio_lane2
            GPIO.output(12,0)
            GPIO.output(10,1)
            time.sleep(2)
            GPIO.output(10,0)
            GPIO.output(8,1) 
            time.sleep(t[1])
            GPIO.output(8,0)             
            GPIO.output(10,1)            
            time.sleep(2)                
            GPIO.output(10,0)
            GPIO.output(12,1)
    if(4==len(emergency)):
        if(emergency[2] == 1):
            print("Emergency Vehicle Detected in Lane 3")
            #gpio_lane3
            GPIO.output(15,0)
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0) 
            GPIO.output(11,1) 
            time.sleep(t[2])
            GPIO.output(11,0)             
            GPIO.output(13,1)            
            time.sleep(2)                
            GPIO.output(13,0)                       
            GPIO.output(15,1)
    maxlimit.release()'''
    lock.release()
@with_goto
def itms(lockg,emergency,count):
    lockg.acquire()
    print("GPIO THREADING BEGINS SOOOON")
    print("-----------------------------------------------------------------")
    # assign timings to array t
    label .c
    if(4==len(count)):
        c=count
        t = []
        for l in range(4):
            ti = (c[l] * 4) + 2
            t.append(ti)
        now = datetime.now()
        three=threading.active_count()
        print("")
        print("Current time =", now)
        print("")
        table = PrettyTable([' ','Lane 1','Lane 2','Lane 3','Lane 4'])
        table.add_row(['Vehicle Count', c[0],c[1],c[2],c[3]] )
        table.add_row(['Green Signal Time', t[0],t[1],t[2],t[3]] )
        print(table)
        print("")
        print("Total Active Thread = ", three)
        print("")
        if maxlimit.acquire():
            z = threading.Thread(target=gpio, args=(lock,emergency,t))
            z.start()
    else:
        goto .c
    lockg.release()
def density():
    print("LANE VEHICLE COUNTING BEGUN")
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)   
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 5)]  # adjust range for # of images in folder
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            count = []
            emergency = []
            i = 1
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                labels = []
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)               
                classes = np.squeeze(classes).astype(np.int32)
                def count_vehicle(scores):
                    final_score = np.squeeze(scores)
                    c = 0   
                    for j in range(100):
                        if scores is None or final_score[j] > 0.5:
                                c = c + 1 #taking count of vehicles
                    return c
                def em(scores):
                    e = 0
                    final_score = np.squeeze(scores)
                    for j in range(100):
                        if scores is None or final_score[j] > 0.5:
                            if classes[j] in category_index.keys():
                                labels.append(category_index[classes[j]]['name'])
                    if 'Emergency-AMBULANCE' in labels:
                        e = 1
                    else:
                        e = 0
                    return e  
                #plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)    # matplotlib is configured for command line only so we save the outputs instead
                plt.savefig("outputs/detection_output{}.png".format(i))  # create an outputs folder for the images to be saved          
                count.append(count_vehicle(scores))
                emergency.append(em(scores))
                i = i+1  # this was a quick fix for iteration, create a pull request if you'd like
    y=threading.Thread(target=itms(lock,emergency,count))
    y.start()
    density()
if __name__ ==  '__main__':
    x=Process(target=density)
    x.start()