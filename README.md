# OpenCV Facial Filters

![bleh](https://user-images.githubusercontent.com/29514438/50378212-d01a9a00-0652-11e9-8d45-de6ec1f13dd8.PNG)

## Overview
This project demonstrates the creation of Snapchat-like facial filters using Deep Learning and OpenCV. It involves facial keypoint detection to superimpose themed filters on a face in real time. I came up with the idea for the project on 21 December 2018, Hence the Christmas theme :P.

Detailed methodology and insights can be found in [this Medium article](https://medium.com/@rohit_agrawal/implementing-snapchat-like-filters-using-deep-learning-13551940b174).

## Dataset
Utilized the [Facial Keypoints Detection dataset](https://www.kaggle.com/c/facial-keypoints-detection/data) from Kaggle.

## Methodology
- **Data Augmentation:** Flipped images and key points for diversity.
- **Model Architecture:**
    - **CNN:** Acts as a feature extractor.
    - **ANN:** A fully connected network for facial keypoint regression.
- **Training:**
    - **Loss Metric:** Mean Absolute Loss.
    - **Performance:** Achieved ~0.0113 after 300 epochs with the Adam optimizer.
- **Implementation:**
    - **Real-time Data Capture:** Used OpenCV for live webcam feed.
    - **Preprocessing:** Standardized input before feeding into the model.
    - **Output Utilization:** Keypoint positions determined the placement and scale of thematic filters.
