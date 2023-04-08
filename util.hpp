//
//  util.hpp
//  Project2
//
//  Created by Thean Cheat Lim on 2/7/23.
//

#ifndef util_hpp
#define util_hpp
#include <opencv2/opencv.hpp>

// Return the input number but clamp/limit the value to be within [lower, upper]
// input - input number
// lower - lower bound
// upper - upper bound
int clamp(int input, int lower, int upper);

// Return distance =  sum of squared differences between x and y
// x - a vector of float numbers
// y - another vector of float numbers
float sumSquared(std::vector<float> x, std::vector<float> y);

// Return distance =  1 - normalized histogram intersection between x and y
// x - a vector of float numbers
// y - another vector of float numbers
float histIntersectionNormalized(std::vector<float> x, std::vector<float> y);

// Filters for SobelMagnitude
// Apply a 3x3 Sobel filter (X direction) onto the source image
// src - Source image
// dst - Destination image
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Apply a 3x3 Sobel filter (Y direction) onto the source image
// src - Source image
// dst - Destination image
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Compute the Sobel Magnitude.
// sx - Image with `sobelX3x3` applied
// sy - Image with `sobelY3x3` applied
// dst - Destination image
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
#endif /* util_hpp */
