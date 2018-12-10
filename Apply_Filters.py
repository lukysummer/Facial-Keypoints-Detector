import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from glob import glob


###############################################################################
########## 1. Create model instance & Load optimal trained weights ############
###############################################################################
from model import face_keypoints_network
   
# load the trained Face_Keypoints_Dection model's optimal weights to the CNN
final_model = face_keypoints_network()
final_model.load_state_dict(torch.load('keypoints_network_final.pt', map_location='cpu'))
# Uncomment if using GPU
# final_model.cuda()
# final_model.load_state_dict(torch.load('keypoints_network_final.pt'))
final_model.eval()


###############################################################################
################# 2. Load the test image & detect faces in it #################
###############################################################################
img_path = "images/mona_lisa.jpg"
images = np.array(glob(img_path))
img_h, img_w = plt.imread(img_path).shape[1], plt.imread(img_path).shape[0]


###############################################################################
################# 3. Predict keypoints on the detected faces ##################
###############################################################################
from torch.autograd import Variable  

faces, keypoints = [], []

def detect_keypoints(path):   
    # 1. Read in the human face image & save it in 'faces' array as an RGB image
    face = cv2.imread(path)
    face_RGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_RGB = cv2.resize(face_RGB, (224, 224))

    # 2. Convert the face region from RGB to Grayscale:
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    
    # 3. Normalize the grayscale image s.t. its color range falls in [0,1], not [0,255]:
    face = (face/255.).astype(np.float32)
    
    # 4. Rescale the image to [224, 224]:
    face = cv2.resize(face, (224, 224))
    
    # 5. Resahpe the numpy image shape [H x W x C] into torach image shape [C x H x W]:
    if(len(face.shape) == 2):
        face = np.expand_dims(face, axis = 0)
    else:
        face = np.transpose(face, (2, 1, 0))
    
    # 6. Make it a batch of length 1:
    face = np.expand_dims(face, axis = 0)
    
    # 7. Make it a Torch Tensor & into a Variable:
    face = torch.from_numpy(face)
    face = Variable(face)
    face = face.type(torch.FloatTensor)
    # Uncomment if using GPU
    # face = face.cuda()
    
    # 8. Perform a forward pass to get the predicted facial keypoints
    output = final_model.forward(face)
    output = output.view(output.size()[0], 68, -1)
    
    # 9. Transform the predicted keypoints from troch tensor to numpy array & UNnormalize it:
    #    (since facial keypoints detection training points were normalized with mean = 100 & std = 50, 
    #    the output of the final_model was already normalized)
    pred = output[0].cpu().data
    pred = pred.numpy()
    pred = pred * 50 + 100
    
    return face_RGB, pred


###############################################################################
######################### 4. Apply Dog Ears Filter ############################
############################################################################### 
from filters import Dog_Ears

img, keypoints = detect_keypoints(img_path)    
dog_filter = Dog_Ears('images/dog.png')
cool_dog = dog_filter(img, keypoints)
cool_dog = cv2.resize(cool_dog, (img_h, img_w))


###############################################################################
###################### 5. Visualize the filtered image ########################
###############################################################################
plt.figure(figsize = (6, 6))
plt.imshow(cool_dog)
plt.axis('off')
plt.show()