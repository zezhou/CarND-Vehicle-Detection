## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/zezhou/CarND-Vehicle-Detection/blob/master/writeup.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 6th code cell of the [IPython notebook](https://github.com/zezhou/CarND-Vehicle-Detection/blob/master/p5.ipynb).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

[car_nocar_example]: ./output_images/car_nocar_example.png
![alt text][car_nocar_example]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are two examples using HOG parameters of `orient = 9`, `pix_per_cell = 8` and `cell_per_block = 2`:

[car_hog]: ./output_images/car_hog.png
![alt text][car_hog]

[nocar_hog]: ./output_images/nocar_hog.png
![alt text][nocar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that  `orient = 11` , `pix_per_cell = 16`, `cell_per_block = 2` and `hog_channel = "ALL"` have the highest score, which is `0.9831`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial, `YUV` color features and all channel HOG features. Finally I get a 0.9831 accuracy model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the 47th code cell of the IPython notebook.

I found that some window size was very effective for some car cases, but insensitive to other cases. That is to say, different window size and overlap rates have different effects on car-finding predicting results. So I decide to use various size of sliding windows and overlaps.

Here is a example of sliding windows with parameters `xy_overlap=(0.5, 0.5)` and `xy_windows = (64, 64)`

[sliding_window1]: ./output_images/sliding_window1.png
![alt text][sliding_window1]

Here is another example of sliding windows with parameters `xy_overlap=(0.5, 0.5)` , `x_start_stop=[400, None]`, `y_start_stop=[400, 500]` , and `xy_windows = (100, 100)`

[sliding_window2]: ./output_images/sliding_window2.png
![alt text][sliding_window2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


[feature_result]: ./output_images/feature_result.png
.png
![alt text][feature_result]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

[frames_heatmap]: ./output_images/frames_heatmap.png
![alt text][frames_heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

[frames_bound_label]: ./output_images/frames_bound_label.png
![alt text][frames_bound_label]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

[frames_bound]: ./output_images/frames_bound.png
![alt text][frames_bound]

### Here is the output of bounding boxes with lanes.

[bound_with_lane]: ./output_images/bound_with_lane.png
![alt text][bound_with_lane]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First, it caused memory error when I trained the car finding model. This is because my laptop's memory is only 4GB, which was not enough when `spatial_size = (64, 64)`. To resolve this problem, I change `spatial_size = (16, 16)` and the training was succeed.

Second, as course mentioned, matplotlib image will read .jpg images in on a scale of 0 to 255 so if you are testing your pipeline on .jpg images remember to scale them accordingly. 

Third, the key of improving accuracy of finding car is using various size of sliding windows in the pipeline. For example, when I use default size of sliding windows (64, 64), the predicting results lost a lot of true positive results. 

[window_size1]: ./output_images/window_size1.png
![alt text][window_size1]

when I changed size of sliding windows to (128,128) , I get a better result.

[window_size2]: ./output_images/window_size2.png
![alt text][window_size2]

Finally, I use 5 diffrent size of sliding windows in the  pipeline function. 

The last but not the least, you can cut the sliding window region to accelarate the processing time. For example, I set `x_start_stop=[400, None]` and `y_start_stop=[400, 500]` in some cases in pipeline.