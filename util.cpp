//
//  util.cpp
//  Project2
//
//  Created by Thean Cheat Lim on 2/7/23.
//

#include "util.hpp"

// Return the input number but clamp/limit the value to be within [lower, upper]
// input - input number
// lower - lower bound
// upper - upper bound
int clamp(int input, int lower, int upper){
    if (input < lower) return lower;
    if (input > upper) return upper;
    return input;
}

// Return distance =  sum of squared differences between x and y
// x - a vector of float numbers
// y - another vector of float numbers
float sumSquared(std::vector<float> x, std::vector<float> y){
    float result = 0.0f;
    for (std::size_t i = 0; i < x.size(); i++){
        result += (x[i] - y[i])*(x[i] - y[i]);
    }
    return result;
}

// Return distance =  1 - normalized histogram intersection between x and y
// x - a vector of float numbers
// y - another vector of float numbers
float histIntersectionNormalized(std::vector<float> x, std::vector<float> y){
    float result = 0.0f;
    for (std::size_t i = 0; i < x.size(); i++){
        result += std::min(x[i], y[i]);
    }
    return 1-result;
}

// Apply a 3x3 Sobel filter (X direction) onto the source image
// src - Source image
// dst - Destination image
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
    // Positive Right
    //    [-1, 0, 1]
    //    [-2, 0, 2]
    //    [-1, 0, 1]
    //
    dst = cv::Mat::zeros(src.size(), CV_16SC3); // signed short data type
    // Row 1D
    // [-1, 0, 1]
    for(int i=0; i<src.rows; i++){
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        // Destination pointer
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
        for(int j=1; j<src.cols-1; j++){
            for(int c=0;c<3;c++){
                dptr[j][c] =
                (
                 -1*sptr[j-1][c]
                 +1*sptr[j+1][c]
                );
            }
        }
    }
    
    cv::Mat temp;
    dst.copyTo(temp);
    
    // Column 1D
    //    [1]
    //    [2]
    //    [1]
    for(int j=0; j<src.cols; j++){
        for(int i=1; i<src.rows-1; i++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i, j)[c] =
                (
                 1*temp.at<cv::Vec3s>(i-1, j)[c]
                 +2*temp.at<cv::Vec3s>(i, j)[c]
                 +1*temp.at<cv::Vec3s>(i+1, j)[c]
                 )/4;
            }
        }
    }
    
     return 0;
}

// Apply a 3x3 Sobel filter (Y direction) onto the source image
// src - Source image
// dst - Destination image
int sobelY3x3(cv::Mat &src, cv::Mat &dst){
    // Positive Up
    //  [ 1 2 1]
    //  [ 0 0 0]
    //  [ -1 -2 -1]
    dst = cv::Mat::zeros(src.size(), CV_16SC3); // signed short data type
    
    // Row 1D
    // [1, 2, 1]
    for(int i=0; i<src.rows; i++){
        cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i);
        // Destination pointer
        cv::Vec3s *dptr = dst.ptr<cv::Vec3s>(i);
        for(int j=1; j<src.cols-1; j++){
            for(int c=0;c<3;c++){
                dptr[j][c] =
                (
                 1*sptr[j-1][c]
                 +2*sptr[j][c]
                 +1*sptr[j+1][c]
                )/4;
            }
        }
    }
    
    cv::Mat temp;
    dst.copyTo(temp);
    // Column 1D
    //    [+1]
    //    [0]
    //    [-1]
    for(int j=0; j<src.cols; j++){
        for(int i=1; i<src.rows-1; i++){
            for(int c=0;c<3;c++){
                dst.at<cv::Vec3s>(i, j)[c] =
                (
                 1*temp.at<cv::Vec3s>(i-1, j)[c]
                 -1*temp.at<cv::Vec3s>(i+1, j)[c]
                 );
            }
        }
    }
     return 0;
}

// Compute the Sobel Magnitude
// sx - Image with `sobelX3x3` applied
// sy - Image with `sobelY3x3` applied
// dst - Destination image
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
    dst = cv::Mat::zeros(sx.size(), CV_8UC3); // unsigned short data type
    for(int i=0;i<sx.rows;i++){
        // sx, sy pointers
        cv::Vec3s *sxptr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *syptr = sy.ptr<cv::Vec3s>(i);
        // destination pointer
        cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
        
        for(int j=0;j<sx.cols;j++){
            for(int c=0;c<3;c++){
                dptr[j][c] = sqrt(pow(sxptr[j][c], 2)+pow(syptr[j][c], 2));
            }
       }
     }
    return 0;
}
