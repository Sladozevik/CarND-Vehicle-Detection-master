
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.jpg
[image2]: ./examples/hog_cars.jpg
[image3]: ./examples/hog_non_cars.jpg
[image4]: ./examples/cars_histogram.jpg
[image5]: ./examples/non_cars_histogram.jpg
[image6]: ./examples/normalized_features.jpg
[image7]: ./examples/example_image.jpg
[image8]: ./examples/example_image_windows.jpg
[image9]: ./examples/slide_windows.jpg
[image10]: ./examples/pipeline_image.jpg
[image11]: ./examples/heatmap.jpg
[image12]: ./examples/pipeline.jpg
[video1]: ./test_video_out_2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. This is writeup that includes all the rubric points and how i have addressed each one.

#### Import Modules, Load images and show images that we will work on

As a first step in cell 1 to 3 I have imported modules for project, loaded images from folders and showed randomly 10 images from Cars and Non Cars data set. 
Here is example of Cars and Non Cars images:

![alt text][image1]

Number of Car images 8792

Number of Non-Car images 8968

### Histogram of Oriented Gradients (HOG)

#### 2. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cells 4 to 6 of the IPython notebook.

First u created function `get_hog_features` and after i i used it to read all the `cars` and `non-cars` images.  Here is an example of one of each of the `cars` and `non_cars` classes:

![alt text][image2]

![alt text][image3]

### Color Histogram 

In cells 7 I have built functions `color_histmg` and `bin_spatial`. `color_histmg` is to extract histogram of color channels and  `bin_spatial` to to compute binned color features.
in cells 8 and 9 i have shown Cars and Non Cars histogram:
 
![alt text][image4]

![alt text][image5]

### Extract Features

In cells 10 - 13 i have built function `extract_features` 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is example of normalized features of image using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:

![alt text][image6]

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using parameters in cell 14.

```color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11 # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
```

While training linear SVM i have tested parameters tuning to get best results. Final parameters that showed **98,96%** Test Accuracy are in above cell.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Cell 15:** in this cell i have created function `draw boxes` and `slide_window` that that takes an image  start and stop positions in both x and y, window size (x and y dimensions),  and overlap fraction (for both x and y)

**Cell 16:** Random example image that will be used to slide window search

![alt text][image7]

**Cell 17:** Example of Slide all windows over image.

![alt text][image8]

**Cell 18 - 19:** defined functions `single_img_features` and `search_windows` that extract features from a single image window. This function is very similar to extract_features() just for a single image rather than list of images. Here is example of search window and as you can see predictions are not that great. 

![alt text][image9]

**Cell 20:** I have created function `find_cars`. This single function that can extract features using hog sub-sampling and make predictions

I have tested function `find_cars` with windows search area of:
```
ystart = 400
ystop = 656  
```
I have also used for loop to use scale of search window from 1 to 2.0 with step of 0.1.
```
for i in np.arange(1,2.0,0.1):
    scale = i
    out_img, recktangles = find_cars(img, ystart, ystop, scale, svc, X_scaler, 
                    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    recktangles_test += recktangles
```

Here is result of above code

![alt text][image10]

**Cell 20:** Heatmap

In cells 22-23 I built a heat-map from slide window detections in order to combine overlapping detections and remove false positives.

![alt text][image11]

### Pipeline

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image12]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video_out_2.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

