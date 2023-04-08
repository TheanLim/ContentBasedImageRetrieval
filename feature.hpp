//
//  feature.hpp
//  Project2
//
//  This files contains functions that can be used to extract features from an input image.
//  Created by Thean Cheat Lim on 2/4/23.
//

#ifndef feature_hpp
#define feature_hpp

#include <opencv2/opencv.hpp>

// Extract the widthxheight pixels from the middle of input image
// and write those pixel values into outputVector
// For each pixel, write the values in the order of B, G and R
// img - Input image
// width - width
// height - height
// outputVector - vector containing features of the input image
int extractMiddleVector(cv::Mat &img, int width, int height, std::vector<float> &outputVector);

// Given an input image, and number of histogram bins,
// create a 3D histogram with `bins` bins, and project each pixel from the input image to the histogram.
// The 3D histogram is normalized.
// Write the 3D histogram into the provided output vector.
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extract3DHistVector(cv::Mat &img, int bins, std::vector<float> &outputVector);

// Given an input image, convert it into Grayscale, compute the Sobel Magnitude,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractSobelTextureVector(cv::Mat &img, int bins, std::vector<float> &outputVector);

// Given an input image, number of histogram bins, and softWidth (width to spread out a pixel value)
// create a 3D soft histogram with `bins` bins, and project and spread each pixel into width of `softWidth`
// from the input image to the histogram.
// The 3D histogram is normalized.
// Write the 3D soft histogram into the provided output vector.
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extract3DSoftHistVector(cv::Mat &img, int bins, int softWidth, std::vector<float> &outputVector);

// Given an input image, convert it into Grayscale, compute and average the 14 Law's Filters output,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractLawsTextureVector(cv::Mat &img, int bins, std::vector<float> &outputVector);

// Given an input image, convert it into Grayscale, apply the Gabor's Filters,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractGaborTextureVector(cv::Mat &src, int bins, std::vector<float> &outputVector);

#endif /* feature_hpp */
