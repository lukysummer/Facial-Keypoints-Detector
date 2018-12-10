import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
import torch.nn.functional as F


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

test_image = cv2.imread('images/effy.png')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(test_image, 1.2, 2)
print('\n', len(faces), ' face(s) detected! :)')

     
    
###############################################################################
################# 3. Predict keypoints on the detected faces ##################
###############################################################################
    
image_copy = np.copy(test_image)
images, keypoints = [], []
pad = 50    # margin outside detected face boundary
img_h, img_w = image_copy.shape[0], image_copy.shape[1]

# loop over the detected faces 
for (x, y, w, h) in faces:
    # 1. Select the region of interest that is the face in the image 
    roi = image_copy[max(0, y-pad): min(y+h+pad, img_h), max(0, x-pad): min(x+w+pad, img_w)]

    ## 2. Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = (roi/255.).astype(np.float32)

    ## 3. Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    try:
        roi = cv2.resize(roi, (224, 224))
    except:
        pass
    ## 4. Append the resulting face into images array
    images.append(roi)
    
    ## 5. Convert the image into grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    ## 6. Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    if len(roi.shape) == 2:
        roi = np.expand_dims(roi, axis=0)
    else:
        roi = np.rollaxis(roi, 2, 0)
    
    ## 7. Make it a batch of length 1
    roi = np.expand_dims(roi, axis=0)
    
    ## 8. Convert the face region image into a torch tensor 
    roi = torch.from_numpy(roi).type(torch.FloatTensor)
    
    ## 9. Pass the image through the final model network
    results = final_model.forward(roi)
    
    ## 10. Reshape the keypoints output into 2 columns & put it in keypoints array
    results = results.view(results.size()[0], 68, -1).cpu()
    
    ## 11. Move the keypoints to cpu & de-normalize the values
    pred = results[0].cpu().data
    pred = pred.numpy()
    pred = pred * 50 + 100

    keypoints.append(pred)

    
###############################################################################
############## 4. Visualize the predicted kepoints on the face(s) #############
###############################################################################
 
def visualize_output(faces, test_outputs):  
    for i, face in enumerate(faces):
        plt.figure(figsize=(5, 5))
        plt.imshow(face)
        plt.scatter(test_outputs[i][:, 0], test_outputs[i][:, 1], s=20, marker='.', c='m')
        plt.axis('off')

    plt.show()


visualize_output(images, keypoints)