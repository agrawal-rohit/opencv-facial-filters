# OpenCV facial filters

A selfie filter implemented using Deep Learning and OpenCV.
Full documentation: https://medium.com/@rohit_agrawal/implementing-snapchat-like-filters-using-deep-learning-13551940b174

**Dataset:** https://www.kaggle.com/c/facial-keypoints-detection/data

**Methodology**

Data from the dataset was augmented by flipping the images and their keypoints. I then used a CNN as the feature extractor, flattened the outputs and passed them into a fully connected ANN to perform facial keypoint regression. Metric used was 'Mean Absolute Loss', the model's best was ~0.0113 after being trained for 300 epochs using adam optimizer.

Once the model was complete, I used OpenCV to get live data from the webcam for real-time predictions. The input was preprocessed and input into the model and the outputs plotted. Then I used the positions of specific keypoints for the position and scale of 'filters' on top of the image. 

I came up with the idea for the project on 21 December 2018, Hence the Christmas theme :P.

**Sample Output**

![bleh](https://user-images.githubusercontent.com/29514438/50378212-d01a9a00-0652-11e9-8d45-de6ec1f13dd8.PNG)
