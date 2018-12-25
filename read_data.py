# Read data

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.io import imshow
import math

# Check if row has any NaN values 
def has_nan(keypoints):
    for i in range(len(keypoints)):
        if math.isnan(keypoints[i]):
            return True
    return False

# Function which plots an image with it's corresponding keypoints
def visualize_points(img, points):
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    imshow(img)
    for i in range(0,len(points),2):
        x_renorm = (points[i]+0.5)*96       # Denormalize x-coordinate
        y_renorm = (points[i+1]+0.5)*96     # Denormalize y-coordinate
        circ = Circle((x_renorm, y_renorm),1, color='r')    # Plot the keypoints at the x and y coordinates
        ax.add_patch(circ)
    plt.show()

# Read the data as Dataframes
training = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

# Get training data
imgs_train = []
points_train = []
for i in range(len(training)):
    points = training.iloc[i,:-1]
    if has_nan(points) is False:
        test_image = training.iloc[i,-1]        # Get the image data
        test_image = np.array(test_image.split(' ')).astype(int)    
        test_image = np.reshape(test_image, (96,96))        # Reshape into an array of size 96x96
        test_image = test_image/255         # Normalize image
        imgs_train.append(test_image)
        
        keypoints = training.iloc[i,:-1].astype(int).values
        keypoints = keypoints/96 - 0.5  # Normalize keypoint coordinates
        points_train.append(keypoints)

imgs_train = np.array(imgs_train)    
points_train = np.array(points_train)

# Get test data
imgs_test = []
for i in range(len(test)):
    test_image = test.iloc[i,-1]        # Get the image data
    test_image = np.array(test_image.split(' ')).astype(int)
    test_image = np.reshape(test_image, (96,96))        # Reshape into an array of size 96x96
    test_image = test_image/255     # Normalize image
    imgs_test.append(test_image)
    
imgs_test = np.array(imgs_test)

# Data Augmentation by mirroring the images
def augment(img, points):
    f_img = img[:, ::-1]        # Mirror the image
    for i in range(0,len(points),2):        # Mirror the key point coordinates
        x_renorm = (points[i]+0.5)*96       # Denormalize x-coordinate
        dx = x_renorm - 48          # Get distance to midpoint
        x_renorm_flipped = x_renorm - 2*dx      
        points[i] = x_renorm_flipped/96 - 0.5       # Normalize x-coordinate
    return f_img, points

aug_imgs_train = []
aug_points_train = []
for i, img in enumerate(imgs_train):
    f_img, f_points = augment(img, points_train[i])
    aug_imgs_train.append(f_img)
    aug_points_train.append(f_points)
    
aug_imgs_train = np.array(aug_imgs_train)
aug_points_train = np.array(aug_points_train)

# Combine the original data and augmented data
imgs_total = np.concatenate((imgs_train, aug_imgs_train), axis=0)       
points_total = np.concatenate((points_train, aug_points_train), axis=0)

def get_train_data():
    imgs_total_reshaped = np.reshape(imgs_total, (imgs_total.shape[0],imgs_total.shape[1],imgs_total.shape[2], 1))
    return imgs_total_reshaped,points_total

def get_test_data():
    return imgs_test