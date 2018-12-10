''' CODE UP UNTIL TRAINING THE NETWORK '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import utils


key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')
use_cuda = torch.cuda.is_available()

###############################################################################
############# 1. LOAD THE TRAINING SET DATA ################
###############################################################################

class Facial_Keypoints_Dataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform = None):
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir # root dir for images
        self.transform = transform
        
    def __len__(self):
        return len(self.key_pts_frame)
     
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])
        image = mpimg.imread(img_path) # numpy array
        if image.shape[2] == 4:
            image = image[:, :, :3] # get rid of alpha color channel
            
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'keypoints': key_pts}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    
face_dataset = Facial_Keypoints_Dataset(csv_file = 'data/training_frames_keypoints.csv',
                                        root_dir = 'data/training')

print('Length of Training Dataset: ', len(face_dataset))



###############################################################################
############ 2. VISUALIZE SOME TRAINING FACE IMAGES W/ KEY POINTS #############
###############################################################################

def show_keypoints(image, key_pts):
    plt.imshow(image)
    plt.scatter(key_pts[:,0], key_pts[:,1], s = 20, marker='.', c = 'm')
    plt.show()
    

num_to_display = 3
for rand_idx in np.random.randint(0, len(face_dataset), num_to_display):
    print(rand_idx)
    sample = face_dataset[rand_idx]
    show_keypoints(sample['image'], sample['keypoints'])
    


###############################################################################
################ 3. DEFINE TRANSFORMS TO BE APPLIED TO IMAGES #################
###############################################################################
    
from torchvision import transforms

class Normalize(object):
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image/255.0
        
        key_pts = (key_pts - 100)/50.0 # assume: mean = 100, std = 50
        return {'image': image, 'keypoints': key_pts}


class Resize(object):
    # Arg: Desired output size 
    #   --> if tupple : returns output of the tuple size
    #   --> if int : Smaller Edge = given int
    #                Longer Edge = given int * (orig smaller edge / orig longer edge)
    #                (keeps the original ratio the same)
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        
        if(isinstance(self.output_size, tuple)):
            new_h, new_w = self.output_size
        else:
            if(h <= w):
                new_h, new_w = self.output_size, self.output_size*(w/h)
            else:
                new_h, new_w = self.output_size*(h/w), self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        new_image = cv2.resize(image, (new_w, new_h))
        #  "*" : [np.arr1 * np.arr2] :
        #           (1) Both arrays have SAME dimension: ELEMENTWISE MULTIPLICAION 
        #           (2) Same # of columns & 1 array has 1 row :  
        #               ELEMENTWISE MULTIPLICAION w/ that 1 row with every row of the longer arr
        
        new_key_pts = key_pts * [new_w/w, new_h/h]
        
        return { 'image' : new_image, 'keypoints' : new_key_pts }   
    
            
    
class RandomCrop(object):
    # Arg: Desired output size 
    #   --> if tupple : returns output of the tuple size
    #   --> if int : returns output of square size (int, int)
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top:top + new_h, left:left + new_w]
        
        # Still keeping all 68 facial key points, 
        # while putting them in "negative" coodintes outside the image
        key_pts = key_pts - [left, top] 
        
        return {'image': image, 'keypoints': key_pts}
        
    

class ToTensor(object):
        
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        if(len(image.shape) == 2): # if image has NO 3rd grayscale channel (1)
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # Put grayscale channel at the 1st dimension, because:
        # Numpy image: [H, W, C]  &  Tensor image: [C, H, W]
        image = image.transpose((2, 0, 1)) # make (0, 1, 2) --> (2, 0, 1)
        return {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(key_pts)}


combined = transforms.Compose([Resize(250), RandomCrop(224)])
row_idx = 899
sample = face_dataset[row_idx]

fig = plt.figure()

for i, tr in enumerate([Resize(100), RandomCrop(50), combined]):
    transformed = tr(sample)
    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tr).__name__)
    show_keypoints(transformed['image'], transformed['keypoints'])
    
plt.show()



###############################################################################
############ 4. APPLY TRANSFORMS & LOAD THE TRAINING & TEST DATA ##############
###############################################################################

data_transform = transforms.Compose([Resize(250), RandomCrop(224), Normalize(), ToTensor()])

train_set = Facial_Keypoints_Dataset(csv_file = 'data/training_frames_keypoints.csv',
                                     root_dir = 'data/training',
                                     transform = data_transform)

test_set = Facial_Keypoints_Dataset(csv_file = 'data/test_frames_keypoints.csv',
                                     root_dir = 'data/test',
                                     transform = data_transform)

print('# of training data: ', len(train_set))
print('# of test data: ', len(test_set))

for i in range(5):
    sample = train_set[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, num_workers = 0, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, num_workers = 0, shuffle = True)



###############################################################################
######################### 5. BUILD MY OWN CNN MODEL ###########################
###############################################################################

from torch import nn, optim
import torch.nn.functional as F

class my_CNN(nn.Module):
    
    def __init__(self):
        super(my_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 600)
        self.fc3 = nn.Linear(600, 136)
        
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 112
        x = self.pool(F.relu(self.conv2(x))) # 56
        x = self.pool(F.relu(self.conv3(x))) # 28
        x = self.pool(F.relu(self.conv4(x))) # 14
        x = self.pool(F.relu(self.conv5(x))) # 7
        
        x = x.view(-1, 256 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
    
net = my_CNN()
if use_cuda:
    net = net.cuda()
    
print(net)



###############################################################################
############# 5. TEST THE MODEL ON TEST SET BEFORE IT IS TRAINED ##############
###############################################################################

from torch.autograd import Variable

def net_sample_output():
    for i, sample in enumerate(test_loader):
        images, key_pts = sample['image'], sample['keypoints']
        # wrap the image in a Variable, s.t. net can process it as input & 
        # track how it changes as the image moves through the network
        images = Variable(images)  
        images = images.type(torch.FloatTensor)
        
        output_pts = net(images)
        # reshape to: [(batch size) x 68 x 2]
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)  
        
        if i == 0:  # break after 1st batch is tested
            return images, output_pts, key_pts

        
test_images, pred_keypts, true_keypts = net_sample_output()
# << SHAPES >>
# test_images : (128, 1, 224, 224)
# pred_keypts : (128, 68, 2)
# true_keypts : (128, 68, 2)


def show_all_keypoints(image, predicted_keypts, true_keypts = None):
    plt.imshow(image, cmap = 'gray')
    plt.scatter(predicted_keypts[:,0], predicted_keypts[:,1], s = 20, marker = '.', c = 'm')
    if true_keypts is not None:
        plt.scatter(true_keypts[:,0], true_keypts[:,1], s = 20, marker = '.', c = 'g')


def visualize_output(test_images, pred_keypts, true_keypts = None, batch_size = 10):
    pred_keypts = pred_keypts.cpu()
    for i in range(batch_size):
        plt.figure(figsize = (60, 30))
        ax = plt.subplot(1, batch_size, i+1)
        
        image = test_images[i].data.numpy()
        image = test_images[i]
        image = np.transpose(image, (1, 2, 0))
        
        pred = pred_keypts[i].data
        pred = pred.numpy()
        pred = pred * 50 + 100
        
        if true_keypts is not None:
            true = true_keypts[i]
            true = true * 50 + 100
        
        show_all_keypoints(np.squeeze(image), pred, true)
        #plt.axis('off')
        
    plt.show()
    
    
visualize_output(test_images, pred_keypts, true_keypts)



###############################################################################
########################## 6. DEFINE LOSS & OPTIMIZER #########################
###############################################################################

initial_lr = 0.00001
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, amsgrad=True, weight_decay=0)

def sqrt_lr_scheduler(optimizer, epoch, init_lr = initial_lr, steps = 50):
    # Decay learning rate by square root of the epoch #
    if (epoch % steps) == 0:
        lr = init_lr / np.sqrt(epoch + 1)
        print('Adjusted learning rate: {:.6f}'.format(lr))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return optimizer



###############################################################################
############################# 7. TRAIN THE NETWORK ############################
###############################################################################        

print_every = 40

def train_net(n_epochs, initial_lr = initial_lr):
    global optimizer
    losses = []
    
    net.train()
    for e in range(n_epochs):
        running_loss = 0.0
        optimizer = sqrt_lr_scheduler(optimizer, e, initial_lr)
        
        for batch_i, data in enumerate(train_loader):
            
            images, key_pts = data['image'], data['keypoints']
            key_pts = key_pts.view(key_pts.size(0), -1)  # flatten it to (1, 136)
            images, key_pts = Variable(images), Variable(key_pts)
            images, key_pts = images.type(torch.FloatTensor), key_pts.type(torch.FloatTensor)
            if use_cuda:
                images, key_pts = images.cuda(), key_pts.cuda()
       
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, key_pts)
            running_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            
            if(batch_i % print_every) == 0:
                print('Epoch: {}, Batch: {}, Traininng Loss: {:.6f}'.format(e+1, batch_i+1, running_loss))
                running_loss = 0.0
                
        losses.append(running_loss)
        
    print('Finished Training')
    return losses

n_epochs = 500
losses = train_net(n_epochs)
