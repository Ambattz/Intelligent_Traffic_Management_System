# INTELLINGENT-TRAFFIC-MANAGEMENT-SYSTEM

In the present day to day life, vehicular traffic is increasing all over the world, especially in urban  areas. With inadequate space and funds for the construction of new roads, and the growing  imbalance between traffic demand and transportation resources; it is increasingly obvious that  countries must move beyond the traditional model of just building roads to solve traffic problems.  A traffic signal controller, as the most important part of infrastructure in smart city transportation,  is the main coordinator for the urban traffic flows. We propose an Intelligent Traffic Signaling  System capable of prioritizing congested lanes based on real-time traffic density. The system  works with cameras positioned at every lane of the intersection for the acquisition of images. These images are transmitted to a credit-card sized computer like Raspberry Pi for traffic density  calculation using computer vision techniques.

IMAGE PROCESSING
* Four traffic videos were taken as input.
* Converted the video to frames.
* Resizing and cropping were done on the frame.

TESTING ON PRE-TRAINED MODEL
* Passed the frame to a neural network for object detection.
* Model used is Mobile Net SSDlite-mobilenet-v2-coco
* Filtered out the weak detection using a minimum confidence.
* Assigned timings to each lane based on the vehicle count.

TESTING ON CUSTOM TRAINED OBJECT DETECTION MODEL
* Gathering data
* Image labelling
* Preparation before training the model
* Create label map file
* Download example model
* Train the model
* Export model
* Test the model

GATHERING DATA
*To train a robust classifier, different images having different backgrounds and varying lighting conditions were gathered.

![image](https://user-images.githubusercontent.com/69767685/141911751-8966d860-5b4e-4b21-a710-2f6712ca4187.png)


IMAGE LABELLING
* Dataset labelling is required for the training purpose.
* Installed labelImg for the same.

![image](https://user-images.githubusercontent.com/69767685/141910811-3603f7f8-c3b0-436b-b773-20b1430b1900.png)

PREPARATION BEFORE TRAINING THE MODEL
* create directory data in workspace.
* clone TensorFlow models repository in the workspace and install dependencies.
* Generate the record file required for pipeline configuration.

![image](https://user-images.githubusercontent.com/69767685/141912122-d44d5706-abc8-4459-b4e2-a5f1cd2b39ab.png)

CREATE LABEL MAP FILE
* multiple data items to be trained are added in the label map file.
* Each object has an id and a name.

![image](https://user-images.githubusercontent.com/69767685/141912435-5fe233e5-2ec5-4908-b455-5d9170d9f4e8.png)

DOWNLOAD EXAMPLE MODEL
* Download the example model  ’ SSD MobileNet v2 lite (COCO)’  and extract in workspace home directory.
* Number of classes =  2
* Batch size = 4

TRAIN THE MODEL
* Train the model using the data gathered to detect objects.
* The model was trained on 50 images.
* Number of training steps = 19000

![image](https://user-images.githubusercontent.com/69767685/141912928-3ad9de40-fc93-42af-8beb-02da8b5d9114.png)

EXPORT THE MODEL
* The below command was ran to export the model

![image](https://user-images.githubusercontent.com/69767685/141913356-9c76fcd7-0f03-4b67-a2f6-f723546697fb.png)

TEST THE MODEL
* Test the newly trained model using new data.

![image](https://user-images.githubusercontent.com/69767685/141914221-81f73aff-13a9-49ed-b4fc-f306a0e09842.png)

SET UP HARDWARE
* Install OS in Raspberry Pi.
* Set up a circuit.
* Install dependencies on Raspberry Pi.

![image](https://user-images.githubusercontent.com/69767685/141914666-602cb000-a575-4eef-8fdc-43f6a3ce339d.png)

TAKE COUNT OF VEHICLES AND ASSIGN TIMING TO EACH LANE
* The number of vehicles on each lane was counted.
* Timing was assigned to each lane based on the count. where, time = count + 1 

INTEGRATE HARDWARE WITH SOFTWARE

![image](https://user-images.githubusercontent.com/69767685/141915135-8e42f4df-afc8-42ce-97f4-d62f33858ffc.png)

MAKE THE TRAFFIC LIGHTS BLINK
* Run the code on Raspberry Pi.
* Make the LED light blink according to the timings assigned

![image](https://user-images.githubusercontent.com/69767685/141915542-b782aeee-c8bb-4d81-84a3-7ec09579c07a.png)

EMERGENCY VEHICLE PREEMPTION 
* Train the model again with images of ambulance.
* The lane where emergency vehicle is detected will get priority

![image](https://user-images.githubusercontent.com/69767685/141916034-4de04125-bc8b-47b2-9c05-0ae87558668d.png)

![image](https://user-images.githubusercontent.com/69767685/141916963-218022a8-ce83-4730-85c7-0696bb98b68c.png)

![image](https://user-images.githubusercontent.com/69767685/141917397-0296f507-2852-4499-a6e2-184f1b9e34c4.png)

![image](https://user-images.githubusercontent.com/69767685/141917861-2e5724a9-2b92-4558-a204-c5c411206903.png)

![image](https://user-images.githubusercontent.com/69767685/141918212-f671ef9c-bd01-4b38-ad9f-46764c11cdad.png)

![image](https://user-images.githubusercontent.com/69767685/141918353-a4d1f569-8be0-4cb1-851e-00b191ea320a.png)

![image](https://user-images.githubusercontent.com/69767685/141918611-45b25324-cdbe-4bda-8c10-a911137767e1.png)

![image](https://user-images.githubusercontent.com/69767685/141918703-a515b4ec-1a74-47b5-bbef-667509970724.png)

![image](https://user-images.githubusercontent.com/69767685/141918806-f9080e4e-eb0d-4f2a-8e11-c200a8aa36ad.png)

![image](https://user-images.githubusercontent.com/69767685/141918932-5ece43bb-dd4e-4253-ae89-f22423e93c86.png)

![image](https://user-images.githubusercontent.com/69767685/141919060-cad5441a-39bd-4ce8-a660-11adb749712b.png)

# RESULT

![image](https://user-images.githubusercontent.com/69767685/141919461-7f92afd4-d1e2-411e-a7a8-e25d2e766569.png)

![image](https://user-images.githubusercontent.com/69767685/141919644-23f6cb2e-9c3c-4b6f-b282-6e3cd0980d26.png)

![image](https://user-images.githubusercontent.com/69767685/141919793-5fe9b199-d07d-43bf-a02c-5d19aa65a019.png)

![image](https://user-images.githubusercontent.com/69767685/141919898-12f304b8-a441-42bd-ae14-037dba810eff.png)

# CONCLUSION

* Human intervention can be significantly  reduced at traffic signals.
* The automated system enables better human  resource management for the police department.
* The idle wait time for commuters can be  greatly reduced, wastage of fuel is avoided, and cut-offs the contribution of traditional system in environment pollution.
* Emergency vehicles are given priority, thus it can be crucial in even saving lives.
* A website that shows live count of vehicles can be beneficial to plan routes ahead.
* The proposed Intelligent Traffic Management System contributes to the economic growth of our country.












