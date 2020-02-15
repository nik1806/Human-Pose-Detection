import cv2
import time
import numpy as np
import math
import glob

#@Attributes -- NkNs,NkRs,NkLs,RsRe,ReRw,LsLe,LeLw,NkRh,RhRn,RnRa,NkLh,LhLn,LnLa,Activity

def angleCal(pointA, pointB):
    x_a, y_a = pointA
    x_b, y_b = pointB
    theta = math.atan2(y_b - y_a, x_b-x_a)
    return theta

protoFile = "model/coco/pose_deploy_linevec.prototxt"
weightsFile = "model/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

if __name__ == "__main__":
    
    action = ['shoot','normal','kick','slap']
    
    for pose in action:
    
        for filename in glob.glob('train_cases/'+pose+'/*'):#iterating over files
            
            frame = cv2.imread(filename)#reading each image
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            threshold = 0.1
            
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            
            t = time.time()
            # input image dimensions for the network
            inWidth = 368
            inHeight = 368
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                      (0, 0, 0), swapRB=False, crop=False)
            
            net.setInput(inpBlob)
            
            output = net.forward()
            print("time taken by network : {:.3f}".format(time.time() - t))
            
            H = output.shape[2]
            W = output.shape[3]
            
            # Empty list to store the detected keypoints
            points = []
            
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
            
                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H
            
                if prob > threshold : 
                    cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                else :
                    points.append(None)
            
                #Opening file to write orientation data
            # file = open('../orient_train.csv', 'a+')    
            
            part_cnt=0
            # Draw Skeleton
            for pair in POSE_PAIRS:
                part_cnt += 1    
                if part_cnt > 13:#only till major limbs
                    break
                partA = pair[0]
                partB = pair[1]
                
                if points[partA] is not None:
                    x,y = points[partA]
                else:
                    file.write(str(90)+',')
                    continue
                
                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                    cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    angle = angleCal(points[partA], points[partB])
                    angle = round(angle*180/np.pi,2)
                    cv2.putText(frame, "{}".format(angle), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, lineType=cv2.LINE_AA)
                    # file.write(str(angle)+',')
                else:
                    pass
                    # file.write(str(90)+',') #if no pair, considering straight down limb
                    
            # file.write(pose+'\n')    
            # file.close()
            cv2.imshow('Output-Keypoints', cv2.resize(frameCopy,(640,360)))
            cv2.imshow('Output-Skeleton', cv2.resize(frame,(640,360)))
            
            
            #cv2.imwrite('Output-Keypoints.jpg', frameCopy)
            #cv2.imwrite('Output-Skeleton.jpg', frame)
            
            print("Total time taken : {:.3f}".format(time.time() - t))
            
            if(cv2.waitKey(0) == ord('q') & 0xFF):
                break
            cv2.destroyAllWindows()
