import cv2
import time
import numpy as np
import math
import glob
from random import randint
#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import argparse

#@Attributes -- NkNs,NkRs,NkLs,RsRe,ReRw,LsLe,LeLw,NkRh,RhRn,RnRa,NkLh,LhLn,LnLa,Activity

protoFile = "model/coco/pose_deploy_linevec.prototxt"
weightsFile = "model/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

# point >=14 are of no use
#POSE_PAIRS_old = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
#              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
#              [1,0], [0,14], [14,16], [0,15], [15,17],
#              [2,17], [5,16] ]

POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14], [14,16], [0,15], [15,17],[2,17], [5,16]]


# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
#mapIdx_old = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
#          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
#          [47,48], [49,50], [53,54], [51,52], [55,56],
#          [37,38], [45,46]]
          
mapIdx = [[47,48], [31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
           [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,0,255], [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], 
         [255,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


file_train = open('orient_train.csv','r')
content = file_train.read()
lines = content.split('\n')
orientations = []
action_stamp = []
avoid = 0
for line in lines:
    avoid+=1
    if avoid == 1 or avoid ==len(lines):
        continue
    data = line.split(',')
    action_stamp.append(data[-1])
    orientations.append([float(data[i]) for i in range(len(data)-1)])

clf = KNeighborsClassifier(n_neighbors=5,weights='distance')

clf = clf.fit(orientations, action_stamp)

file_train.close()

def angleCal(pointA, pointB):
    x_a, y_a = pointA
    x_b, y_b = pointB
    #theta = (y_b - y_a+1)/(x_b-x_a+1)
    theta = math.atan2((y_b - y_a), x_b-x_a)
    return theta


def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

#def classify(orient, pose):
#    file = open('../orient_train.csv','r')
#    content = file.read()
#    lines = content.split('\n')
#    index=0
#    minS=50000
##    maxInd=-1
#    for line in lines[:-1]:
#        summ = 0
#        #print('Line content',line)
#        index += 1
#        angles = line.split(',')[:-1]
#        #print('orientations', angles)
#        for i in range(len(angles)):
#            summ += abs(orient[i]-float(angles[i]))
# ##       print('comp index='+str(index)+' '+str(summ))
#        if summ < minS:
#            minS = summ
#            label = line.split(',')[-1]
# ##           print('allocated label ' + label)
#    
#    if label == pose:
#        result = 'success'
#    else:
#        result = 'failure'
#    return label, result

def classify(orient, pose):
    clf_input = []
    clf_input.append(orient)
    prediction = clf.predict(clf_input)
    
    label = str(prediction[0])    
    
    if label == pose:
        result = 'success'
    else:
        result = 'failure'
    return label, result   


if __name__ == '__main__':
    #when testing individual change 'act' & filepath in 'glob.glob'
#    action = ['kick','shoot','normal','slap']
#
#    act = ['normal']
#    
#    for pose in act:#change to 'action' when reading whole data at once
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="etc/d_fight.mp4", help="Path to test video")
    arg = parser.parse_args()

    filename = arg.video
    cap = cv2.VideoCapture(filename)
    
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(100)
        print("Wait for the header")
    
    f_cnt = 0
    
    t_strt = time.time()
    
    while(True):
        
        
        flag, image1 = cap.read()
        f_cnt += 1
        
        if f_cnt%4 != 0:
            continue
        
        if flag:
            #for filename in glob.glob('../test_cases/'+pose+'/*'): #use when require multiple
           # for filename in glob.glob('trip3.jpg'):
                
           #     image1 = cv2.imread(filename)    
                
            #image1 = cv2.resize(image1, (320,640))            
            #cv2.imshow('original', image1)
            frameWidth = image1.shape[1]
            frameHeight = image1.shape[0]
            
            t = time.time()
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            
            # Fix the input Height and get the width according to the Aspect Ratio
            inHeight = 368
            inWidth = int((inHeight/frameHeight)*frameWidth)
            
            inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                      (0, 0, 0), swapRB=False, crop=False)
            
            net.setInput(inpBlob)
            output = net.forward()
            print("Time Taken in forward pass = {}".format(time.time() - t))
            
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1
            
            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1
            
                detected_keypoints.append(keypoints_with_id)
            
            
            frameClone = image1.copy()
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    ##        cv2.imshow("Keypoints",cv2.resize(frameClone,(480,640)))
            
            valid_pairs, invalid_pairs = getValidPairs(output)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
            
            
            #Opening file to write orientation data
           # file = open('../orient_test_result.csv', 'a+')
            planeimg = image1.copy()
            recognized = ''
            for n in range(len(personwiseKeypoints)):
                orientation = np.ones(13) * 90
                
                for i in range(13):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    angle = (angleCal((B[0], A[0]), (B[1], A[1])))
                    angle = round(angle*180/np.pi,2)
                    orientation[i] = angle
                    cv2.putText(frameClone, "{}".format(angle), (B[0], A[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    #                if i < 4:                
    #                    cv2.putText(frameClone, "{}".format(i), (B[0], A[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2, lineType=cv2.LINE_AA)
    #            
    #                for k in orientation:
    #                    file.write(str(k)+',')
               # pose = 'normal'
                label, result = classify(orientation, 'video')
                
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[1])]
                A = np.int32(keypoints_list[index.astype(int), 0])
                B = np.int32(keypoints_list[index.astype(int), 1])
    
                boxsize = int(1.5*((B[0] - B[1])**2 + (A[0] - A[1])**2)**0.5) 
          #      cv2.imshow("Detected Pose" , frameClone)
                print('********** '+label+' ************')
                
                box_u_lim = max(frameWidth,frameHeight)//10
                
                box_l_lim = max(frameWidth,frameHeight)//14
                
                if boxsize > box_u_lim:
                    boxsize = box_u_lim
                elif boxsize < box_l_lim:
                    boxsize = box_l_lim
                
                if label == 'normal':
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                    cv2.circle(planeimg, (B[0]-0*boxsize, A[0]+0*boxsize), boxsize, color, -2)            
                    recognized = 'Suspicion Alert!!'
                
                #
    #                upper_p = (B[0] - boxsize, A[0] - boxsize)
    #                lower_p = (B[1] + boxsize, A[1] + boxsize)
                #cv2.rectangle(frameClone, upper_p, lower_p, color, 2)
               
    #            file.write(pose +str(',')+str(label))
    #                file.write(str(',')+str(result)) #classifier result
    #               # file.write(pose)
    #                file.write('\n')
    #    
    #            #create a copy and if success then red bounding box
    #            file.close() 
    #       
            print("*********Total time taken = {}".format(time.time() - t))
            print('\n')
            cv2.imshow("Detected Suspicious: " + recognized, planeimg)
            #pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    #        cv2.waitKey(0)
    #        cv2.destroyAllWindows()
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(20)
            print("Waiting for next frame")            
    cap.release()
    cv2.destroyAllWindows()
    print("\n*********Average time taken per image = {}".format((time.time() - t_strt)/(f_cnt/4)))





'''
if __name__ == '__main__':
    #when testing individual change 'act' & filepath in 'glob.glob'
    action = ['kick','shoot','normal','slap']

    act = ['normal']
    
    for pose in action:#change to 'action' when reading whole data at once
    
        for filename in glob.glob('../test_cases/'+pose+'/*'): #use when require multiple
        #for filename in glob.glob('normal.jpeg'):
            
            image1 = cv2.imread(filename)    
            
            image1 = cv2.resize(image1, (320,640))            
            #cv2.imshow('original', image1)
            frameWidth = image1.shape[1]
            frameHeight = image1.shape[0]
            
            t = time.time()
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            
            # Fix the input Height and get the width according to the Aspect Ratio
            inHeight = 368
            inWidth = int((inHeight/frameHeight)*frameWidth)
            
            inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                      (0, 0, 0), swapRB=False, crop=False)
            
            net.setInput(inpBlob)
            output = net.forward()
            print("Time Taken in forward pass = {}".format(time.time() - t))
            
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1
            
            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1
            
                detected_keypoints.append(keypoints_with_id)
            
            
            frameClone = image1.copy()
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    ##        cv2.imshow("Keypoints",cv2.resize(frameClone,(480,640)))
            
            valid_pairs, invalid_pairs = getValidPairs(output)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
            
            
            #Opening file to write orientation data
           # file = open('../orient_test_result.csv', 'a+')
            planeimg = image1.copy()
            for n in range(len(personwiseKeypoints)):
                orientation = np.ones(13) * 90
                for i in range(13):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    angle = (angleCal((B[0], A[0]), (B[1], A[1])))
                    angle = round(angle*180/np.pi,2)
                    orientation[i] = angle
                    cv2.putText(frameClone, "{}".format(angle), (B[0], A[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    #                if i < 4:                
    #                    cv2.putText(frameClone, "{}".format(i), (B[0], A[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 100), 2, lineType=cv2.LINE_AA)
    #            
    #                for k in orientation:
    #                    file.write(str(k)+',')
               # pose = 'normal'
                label, result = classify(orientation, pose)
                cv2.imshow("Detected Pose" , frameClone)
                
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[1])]
                A = np.int32(keypoints_list[index.astype(int), 0])
                B = np.int32(keypoints_list[index.astype(int), 1])
    
                boxsize = int(1.5*((B[0] - B[1])**2 + (A[0] - A[1])**2)**0.5)
                print('********** '+label+' ************')
                if label == 'normal':
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                    if boxsize < 50:
                        boxsize = 30
                
                #boxsize = max(frameWidth,frameHeight)//10
#                upper_p = (B[0] - boxsize, A[0] - boxsize)
#                lower_p = (B[1] + boxsize, A[1] + boxsize)
                #cv2.rectangle(frameClone, upper_p, lower_p, color, 2)
                cv2.circle(planeimg, (B[0]-0*boxsize, A[0]+int(1.2*boxsize)), boxsize, color, -2)            
                
    #            file.write(pose +str(',')+str(label))
    #                file.write(str(',')+str(result)) #classifier result
    #               # file.write(pose)
    #                file.write('\n')
    #    
    #            #create a copy and if success then red bounding box
    #            file.close() 
    #               
            cv2.imshow("Detected Suspicious" , planeimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
'''
