//
//  feature.cpp
//  Project2
//
//  This files contains functions that can be used to extract features from an input image.
//  Created by Thean Cheat Lim on 2/4/23.
//
#include <opencv2/opencv.hpp>
#include "feature.hpp"
#include "util.hpp"

// Extract the widthxheight pixels from the middle of input image and write those pixel values into outputVector
// For each pixel, write the values in the order of B, G and R
// img - Input image
// width - width
// height - height
// outputVector - vector containing features of the input image
int extractMiddleVector(cv::Mat &img, int width, int height, std::vector<float> &outputVector){
    int midRow = (img.rows%2 == 0)? img.rows/2 : img.rows/2+1;
    int midCol = (img.cols%2 == 0)? img.cols/2 : img.cols/2+1;
    
    for(int i=midRow-height/2; i<midRow-height/2+height; i++){
        // Src pointer
        cv::Vec3b *sptr = img.ptr<cv::Vec3b>(i);
        //loop over the columns
        for(int j=midCol-width/2; j<midCol-width/2+width; j++){
            outputVector.push_back(sptr[j][0]);
            outputVector.push_back(sptr[j][1]);
            outputVector.push_back(sptr[j][2]);
        }
    }
    return 0;
}

// Given an input image, and number of histogram bins,
// Create a 3D histogram with `bins` bins, and project each pixel from the input image to the histogram.
// Normalize the histogram
// Write the 3D histogram into the provided output vector.
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extract3DHistVector(cv::Mat &img, int bins, std::vector<float> &outputVector){
    int sizes [] = {bins, bins, bins};
    cv::Mat histMat(3, sizes, CV_32FC1, cv::Scalar(0));

    // Loop through row and col of image
    for(int i=0; i<img.rows; i++){
        // Src pointer
        cv::Vec3b *sptr = img.ptr<cv::Vec3b>(i);
        //loop over the columns
        for(int j=0; j<img.cols; j++){
            uchar bIdx = sptr[j][0]*bins/256;
            uchar gIdx = sptr[j][1]*bins/256;
            uchar rIdx = sptr[j][2]*bins/256;
            histMat.at<float>(bIdx, gIdx, rIdx)+=1;
        }
    }
    
    // loop through and flatten them into std::vector<float>, while normalizing
    float N = img.rows*img.cols;;
    for(int i = 0; i < bins; i++){
        for(int j = 0; j < bins; j++){
            for(int k = 0; k < bins; k++){
                outputVector.push_back(histMat.at<float>(i,j,k)/N);
            }
        }
    }
    return 0;
}

// Given an input image, number of histogram bins, and softWidth (width to spread out a pixel value)
// create a 3D soft histogram with `bins` bins, and project and spread each pixel into width of `softWidth`
// from the input image to the histogram.
// The 3D histogram is normalized.
// Write the 3D soft histogram into the provided output vector.
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extract3DSoftHistVector(cv::Mat &img, int bins, int softWidth, std::vector<float> &outputVector){
    int sizes [] = {bins, bins, bins};
    cv::Mat histMat(3, sizes, CV_32FC1, cv::Scalar(0));

    // Loop through row and col of image
    for(int i=0; i<img.rows; i++){
        // Src pointer
        cv::Vec3b *sptr = img.ptr<cv::Vec3b>(i);
        //loop over the columns
        for(int j=0; j<img.cols; j++){
            for(int w = -softWidth/2; w<-softWidth/2+softWidth; w++){
                uchar bIdx = clamp(sptr[j][0]+w, 0, 255)/softWidth*bins/256;
                uchar gIdx = clamp(sptr[j][1]+w, 0, 255)/softWidth*bins/256;
                uchar rIdx = clamp(sptr[j][2]+w, 0, 255)/softWidth*bins/256;
                histMat.at<float>(bIdx, gIdx, rIdx)+=1;
            }
        }
    }
    
    // loop through and flatten them into std::vector<float>, while normalizing
    float N = img.rows*img.cols;
    for(int i = 0; i < bins; i++){
        for(int j = 0; j < bins; j++){
            for(int k = 0; k < bins; k++){
                outputVector.push_back(histMat.at<float>(i,j,k)/N);
            }
        }
    }
    return 0;
}

// Given an input image, convert it into Grayscale, compute the Sobel Magnitude,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractSobelTextureVector(cv::Mat &img, int bins, std::vector<float> &outputVector){
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Mat sobelGradMagnitude;
    sobelX3x3(img, sobelX);
    sobelY3x3(img, sobelY);
    magnitude(sobelX, sobelY, sobelGradMagnitude);
    return extract3DHistVector(sobelGradMagnitude, bins, outputVector);
}

// Given an input image, convert it into Grayscale, compute and average the 14 Law's Filters output,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractLawsTextureVector(cv::Mat &img, int bins, std::vector<float> &outputVector){
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::Mat L5 = (cv::Mat_<double>(1, 5) << 1, 4, 6, 4, 1);
    L5 = L5 / 16.0;
    cv::Mat E5 = (cv::Mat_<double>(1, 5) << 1, 2, 0, -2, -1);
    cv::Mat S5 = (cv::Mat_<double>(1, 5) << -1, 0, 2, 0, -1);
    cv::Mat W5 = (cv::Mat_<double>(1, 5) << 1, -2, 0, 2, -1);
    cv::Mat R5 = (cv::Mat_<double>(1, 5) << 1, -4, 6, -4, 1);
    
    std::vector<cv::Mat> filters = {L5, E5, S5, W5, R5};
    cv::Mat sum_mat, mean_mat;
//    sum_mat = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
    sum_mat = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    
    for (int i = 0; i<5; i++){
        for(int j = i; j<5; j++){
            cv::Mat filteredImg;
            cv::sepFilter2D(img, filteredImg, -1, filters[i], filters[j].t());
            sum_mat+=filteredImg;
        }
    }
    mean_mat = sum_mat / 14; // 14 filters in total
    cv::Mat output;
    mean_mat.convertTo(output, CV_8UC3, 1.0);
    
    return extract3DHistVector(output, 8, outputVector);
}

// Given an input image, convert it into Grayscale, apply the Gabor's Filters,
// and use it to as the input image. to the extract3DHistVector function
// img - Input image
// bins - number of histogram bins
// outputVector - vector containing features of the input image
int extractGaborTextureVector(cv::Mat &img, int bins,  std::vector<float> &outputVector){
    cv::Mat dst;
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); 
    // Create the filter
    int kernel_size = 20;
    double sigma = 2;
    double lambda = 5;
    double theta = 0;
    double gamma = 2;
    cv::Mat gabor_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambda, gamma, 0, CV_32F);
    
    // Filter the input image
    cv::Mat filtered_image;
    cv::filter2D(img, dst, -1, gabor_kernel);

    // Normalize the filtered image
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
    dst.convertTo(dst, CV_8UC3);

    return extract3DHistVector(dst, 8, outputVector);
}
