import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
whT = 320 # The parameters of the width and the height is set.

confThreshold = 0.5 # The threshold of the confidence is this parameter.
nmsThreshold = 0.3
with open('coco.names','r') as coco:
    coconames = coco.read().rstrip().split('\n') # The space is removed by rstrip.
    

modelConfiguration = 'yolov3-tiny.cfg' #The files are downloaded from the website of the yolo
modelWeights = 'yolov3-tiny.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration,modelWeights) # We created a network as the net. We have installed necessary file.
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # The OpenCV is declared to use in the backend.
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) # The net will be using the CPU.


# print(coconames)
# print(len(coconames))
# hT , wT  = None , None
def findObjects(outputs, img):# The function is created to find objects.
    hT ,wT, cT = img.shape
    bbox =[]  # It will hold features of the objects.
   
    classIds = [] # It will hold what it  is.
    confidences = [] # It will hold confidences value
    
    for output in outputs: # There were 3 outputs.
        for det in output: # It has 85 columns for each output
            scores = det[5:] # Only the part with the objects was taken.
            classId = np.argmax(scores) # The index of the location of the object will hold.
            
            # print(classId) # It will hold what it is.
            
            confidence = scores[classId] # The confidence is found.
            
            if confidence > confThreshold:
                
                w,h = int(det[2]*wT) , int(det[3]*hT) # The width and height of the bounding box is found.
                x1,y1 = int((det[0]*wT)-w/2),int((det[1]*hT)-h/2) # Start point of the bounding box.
                bbox.append([x1,y1,w,h]) # The information is about features of the object.
                classIds.append(classId) # The index of the class is added.
                confidences.append(float(confidence))
    # print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confidences,confThreshold,nmsThreshold) # It is a filter processing (Apply Non Maxima Suppression)
    
    print(indices)
    if(len(indices) > 0):
        for i in indices:
            
            # print(i)
            box = bbox[i] # The object is created.
            x1,y1,w,h = box[0],box[1],box[2],box[3] # The feature of the objects.
            # The rectangle is drawn.The text is written.
            cv.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
            text = coconames[i]
            confidencess = f'{int(confidences[i]*100)}%'
            cv.putText(img, confidencess, ((x1+w-20), y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2) 
            cv.putText(img, text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)    
                
                
                
                
                
while True:
    success, img = cap.read()
    
    blob = cv.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],swapRB = True,crop = False) # The image converted blob
    net.setInput(blob) # The input adjusted  blob.
    
    layerNames = list(net.getLayerNames()) # The layer names are gotten from this module.
    # print(layerNames)
    
   
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()] # The list is used as 1 missing because it started from scratch. Output names are written.
    # print(outputNames)
    
    # print((net.getUnconnectedOutLayers())) # Output layers are written.
    
    outputs = net.forward(outputNames)
    # print(outputs[0].shape) # It has 85 columns.Normally, it has got 80 different objects.Also, there are centerX,centerY,height,width and confidence values.
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0]) # It is only first line.
    # print(type(outputs[0]))
    # print(len(outputs))
    
    
    
    
    findObjects(outputs, img) # The function is called.
    
    if success:
        cv.imshow('Image',img)
        if  cv.waitKey(1) & 0xFF == ord('a'): # It pushes 'a' to exit.
            break
        
        
cv.destroyAllWindows()
cap.release()
