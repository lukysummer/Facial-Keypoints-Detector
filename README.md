# Face Keypoints Detection & Applying Snapchat Filters in PyTorch, CV2

This is my implementation of a face keypoints detection algorithm, which predicts the keypoints of a face as below, and applies Snapchat-like 
Dog Ears filter using the coordinates of the keypoints detected:
<img src="images/result.png">
The main task was to carry out image-to-image translation from Horse to Zebra.

## Repository 

This repository contains:
* **face_keypoints_detection.py** : Complete code for implementing facial keypoints detection using the dataset
* **predict_keypoints.py** : Code for predicting facial keypoints for a new face image
* **filters.py** : Code for applying Dog Ear & Sunglasses filters to a given face image input
* **Apply_Filters.py** : Code for predicting keypoints of a new face image & applying Dog Ear filter to it 
					  
## Datasets

Datasets necessary for this implementation can be downloaded from [this link](https://github.com/udacity/P1_Facial_Keypoints/tree/master/data).

## List of Hyperparameters used:

* Batch Size = **128**
* Generated Image Size = **128 x 128**  
* Number of Convolutional Layers = **5**
* Number of Pooling Layers = **5**
* Number of Fully Connected Layers at the end = **3**
* Initial Learning Rate = **0.00001**
* Number of Epochs = **500**

## Sources

I referenced the following sources for building & debugging the final model :

* https://github.com/udacity/P1_Facial_Keypoints/tree/master



