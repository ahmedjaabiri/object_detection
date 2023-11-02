import cv2
import numpy as np


thres = 0.5 # Threshold to detect object
nms_threshold = 0.3
cap = cv2.VideoCapture('BBAD.mp4')
cap.set(10,1000)
#recording
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
timing = False
def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    if success==True:
        key = cv2.waitKey(1)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])#eliminate parenthese
    confs = list(map(float,confs))#to float

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    for i in indices:
        #extra bracket removed
        i = i[0]
        box = bbox[i]
        confidence=confs[i]
        #x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #resize----------------------------------------------------------------------
    frame_resized = rescaleFrame(img)
    cv2.imshow("Output",frame_resized)
    #screenshot
    if key & 0xFF == ord("s"):
        cv2.imwrite('screenshot'+str(cap.get(cv2.CAP_PROP_POS_FRAMES))+'.jpg',img)
    #fastforward
    if key & 0xFF == ord("f"):
        frnumber=cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frnumber+60)
    #back
    if key & 0xFF == ord("b"):
        frnumber=cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frnumber-60)
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
out.release()
cv2.destroyAllWindows()