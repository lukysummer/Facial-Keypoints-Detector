import imutils
import cv2
import numpy as np


class filters():
    
    def __init__(self, face, keypoints):
        self.face = face
        keypoints = keypoints
        
        
        
class Sunglasses(object):
    
    def __init__(self, img_path):
        self.img_path = img_path
        
        
    def __call__(self, face, keypoints): 
        sunglasses = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        img_cp = np.copy(face)
    
        brow_height_diff = abs(keypoints[18-1, 1] - keypoints[27-1, 1])
        brow_width = abs(keypoints[18-1, 0] - keypoints[27-1, 0])
        angle_brows = np.arctan(brow_height_diff/brow_width)*(180/np.pi)
        angle_brows = angle_brows if keypoints[18-1, 1] < keypoints[27-1, 1] else -angle_brows
        
        rotated_sunglasses = imutils.rotate_bound(sunglasses, angle_brows)
        
        y = int( min(keypoints[18-1, 1], keypoints[27-1, 1]) )
        # Height of sunglasses: length of nose
        h = abs(keypoints[52-1, 1] - keypoints[28-1, 1])  
        # width of sunglasses: distance btwn left & right eyebrows, inclusive
        w = abs((keypoints[17-1, 0]+keypoints[27-1, 0])/2 - 
                (keypoints[18-1, 0]+keypoints[1-1, 0])/2)
        # Rectangular region of the face to put the sunglasses on
        top_nose = keypoints[28-1]
        rectangle_sunglasses = img_cp[y:int(y+h), int(top_nose[0]-w/2):int(top_nose[0]+w/2)]
        
        new_sunglasses = cv2.resize(rotated_sunglasses, (int(w), int(h)), interpolation = cv2.INTER_CUBIC)
        # find all non-transparent points
        ind = np.argwhere(new_sunglasses[:,:,3] > 0) 
        # For each non-transparent points, REPLACE the original image pixel with that 
        # of the new_sunglasses:
        for i in range(3):  # for R, G, B:
            rectangle_sunglasses[ind[:, 0], ind[:, 1], i] = new_sunglasses[ind[:, 0], ind[:, 1], i]
        
        return img_cp
    
    
    
class Dog_Ears(object):
    
    def __init__(self, img_path):
        self.img_path = img_path
        
    
    def __call__(self, face, keypoints): 
        dog = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        img_cp = np.copy(face)
        
        brow_height_diff = abs(keypoints[18-1, 1] - keypoints[27-1, 1])
        brow_width = abs(keypoints[18-1, 0] - keypoints[27-1, 0])
        angle_brows = np.arctan(brow_height_diff/brow_width)*(180/np.pi)
        angle_brows = angle_brows if keypoints[18-1, 1] < keypoints[27-1, 1] else -angle_brows
        
        rotated_dog = imutils.rotate_bound(dog, angle_brows)
        
        brow_to_forhead = int(keypoints[20-1, 1] - abs(keypoints[20-1, 1] - keypoints[9-1, 1])/3.5)
        # x = int(keypoints[1-1, 0])
        h = int(abs(keypoints[20-1, 1] - keypoints[9-1, 1])/2)
        w = int(abs(keypoints[1-1, 0] - keypoints[17-1, 0])*1.3)
        
        top_nose = keypoints[28-1]
        if((int(top_nose[0]-w/2) < 0) or (int(top_nose[0]+w/2) > img_cp.shape[1])):
            print('ERROR! : Please insert a more ZOOMED OUT picture so dog ears can fit! :D')
            return None
        
        else:
            if((brow_to_forhead - h) < 0):
                h = int(h*0.8)
               
            if((brow_to_forhead - h) < 0):
                print('ERROR! : Please insert a more ZOOMED OUT picture so dog ears can fit! :D')   
            else:
                
                rectangle_dog = img_cp[brow_to_forhead - h:brow_to_forhead, int(top_nose[0]-w/2):int(top_nose[0]+w/2)]
               
                new_dog = cv2.resize(rotated_dog, (w, h), interpolation = cv2.INTER_CUBIC)
                
                ind = np.argwhere(new_dog[:,:,3] > 0) # find all non-transparent points
                
                new_dog = cv2.cvtColor(new_dog, cv2.COLOR_BGR2RGB)
                
                for i in range(3):  # for R, G, B:
                    rectangle_dog[ind[:, 0], ind[:, 1], i] = new_dog[ind[:, 0], ind[:, 1], i]
                
                return img_cp
        