import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture('../street.mp4')
whT = 320
confThreshold =0.4
nmsThreshold= 0.3
cap.set(10,1000)
#recording
fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
timing = False
#
def rescaleFrame(frame, scale=0.5):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
 
#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
 
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
 
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, box, (255, 0 , 255), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
 
while True:
    success, img = cap.read()
    if success:
        key=cv.waitKey(1)

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print(outputs[0][0])
    findObjects(outputs,img)
    #resize----------------------------------------------------------------------
    frame_resized = rescaleFrame(img)
    cv.imshow("Output",frame_resized)
    #screenshot
    if key & 0xFF == ord("s"):
        cv2.imwrite('screenshot'+str(cap.get(cv2.CAP_PROP_POS_FRAMES))+'.jpg',img)
    #fastforward
    if key & 0xFF == ord("f"):
        frnumber=cap.get(cv.CAP_PROP_POS_FRAMES)
        cap.set(cv.CAP_PROP_POS_FRAMES, frnumber+60)
    #back
    if key & 0xFF == ord("b"):
        frnumber=cap.get(cv.CAP_PROP_POS_FRAMES)
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    #recording
    if timing:
        out.write(img)
    if key & 0xFF == ord(" "):
        if timing:
            break
        else:
            timing = True
    #
    if key & 0xFF==ord('d'):
        break
cap.release()
cv.destroyAllWindows()