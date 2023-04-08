# ContentBasedImageRetrieval

## Description
This project is about Content-Based Image Retrieval, where a target image is chosen, and the top N matching images are retrieved from an image database. This project explored different ways of creating image features, calculating and combining distances. Some of the features are 3D RGB normalized histograms, soft histograms, and texture filters (Sobel Magnitude, Law’s Filter, and Gobar Filter). The two distance metrics used were: Sum of squared differences and Histogram Intersections.

## Demo

** The leftmost images are the __target images__.

<img src="/images/Middle.png" width="800" height="200">  

**Features**:  9x9 square in the middle of the pic.1016.jpg  
**Distance Metric**: sum-of-squared-difference  

<img src="/images/RGBHistogram.png" width="800" height="200">

**Features**: Top and bottom halves RGBnormalized histograms of the image (pic.0274.jpg), using 8 bins for each RGB  
**Distance Metric**: histogram intersection (equal weights)  

<img src="/images/Sobel.png" width="800" height="200">

**Features**: Whole image RGB normalized histogram and a Sobel Magnitude Grayscale histogram, using 8 bins  
**Distance Metric**: histogram intersection (equal weights)  

## Instructions
- Keep all the provided codes and a reference image directory within the same directory

- Run the following:

	`imgRetrieval.cpp <targetImg path> <image directory path> <featureType> <matchingMethod> <K> <Compute feature vector for the image directory or not>`
	- featureType
		- 1 - the middle 9x9 pixels
		- 2 - whole image 3D Histogram with bins of 8 each
		- 3 - Split image to top and bottom; 3D Histogram with bins of 8 each
		- 4 - whole image 3D Histogram with bins of 8 each + 3D Histogram of Sobel Magnitude with bins of 8 each
		- 5 -  Only use the middle 100x100 and 50x50 pixels; 3D Histogram with bins of 8 each; 3D Histogram of Gobar Filter with bins of 8 each
		- 6 - 3D SOFT Histogram with bins of 8 each, and softWidth of 5
		- 7 - whole image 3D Histogram with bins of 8 each +  3D Histogram on Law's Filter Averaged, with bins of 8 each
		- 8 - 3D Histogram of Sobel Magnitude with bins of 8 each
		- 9 - 3D Histogram on Law's Filter Averaged, with bins of 8 each
		- 10 - 3D Histogram of Gabor's Filter with bins of 8 each
	- matchingMethod aka distance metric
		- 1 - Sum of Square differences
		- 2 - Normalized Histogram intersection distance
	- K - the number of images N to return
	- Compute feature vector for the image directory or not
		- 0 - Don't recompute feature vectors for the images in the image directory
		- 1 - Compute feature vectors for the images in the image directory. Choose this on your first run.
    
## OS and IDE
OS:
MacOS Ventura 13.0.1 (22A400)

IDE:
XCode
