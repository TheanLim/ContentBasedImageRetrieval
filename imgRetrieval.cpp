//
//  imgRetrieval.cpp
//  Project2
//
//  Content-Based Image Retrieval
//  This file contains three functions:
//  createFeatureVector(), knn() and main()
//  Created by Thean Cheat Lim on 2/4/23.
//
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include "feature.hpp"
#include "csv_util.hpp"
#include "util.hpp"

#include <opencv2/features2d.hpp>

// Filenames
char MIDDLE_FEATURE [] = "NineByNine.csv";
char HIST_FEATURE [] = "Hist.csv";
char HIST_UPPERHALF_FEATURE [] = "HistUpperHalf.csv";
char HIST_LOWERHALF_FEATURE [] = "HistLowerHalf.csv";
char HIST_SOBEL_TEXTURE_FEATURE [] = "HistSobelTexture.csv";
char HIST_SOFT_FEATURE [] = "HistSoft.csv";
char HIST_LAWS_FEATURE [] = "HistLaws.csv";
char HIST_GABOR_FEATURE [] = "HistGabor.csv";
char HIST_MIDDLE_MED_FEATURE [] = "HistMiddleMed.csv";
char HIST_MIDDLE_SMALL_FEATURE [] = "HistMiddleSmall.csv";
char HIST_MIDDLE_SMALL_GABOR_FEATURE [] = "HistSmallGabor.csv";
char HIST_MIDDLE_MED_GABOR_FEATURE [] = "HistMiddleGabor.csv";

// Loops through each image from imgDirectory,
// compute feature vectors (according to featureType)
// and store them in csv files.
// imgDir - image Directory
// featureType - Feature type, ranging from 1 to 10
int createFeatureVector(char *imgDir, int featureType){
    // File looping codes from Bruce A. Maxwell
    char dirname[256];
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    // get the directory path
    strcpy(dirname, imgDir);
    printf("Processing directory %s\n", dirname );
    // open the directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
      printf("Cannot open directory %s\n", dirname);
      exit(-1);
    }
    
    // loop over all the files in the image file listing
    int iter = 0;
    while( (dp = readdir(dirp)) != NULL ) {
      // check if the file is an image
      if(
         strstr(dp->d_name, ".jpg") ||
         strstr(dp->d_name, ".png") ||
         strstr(dp->d_name, ".ppm") ||
         strstr(dp->d_name, ".tif")
         )
      {
          printf("processing image file: %s\n", dp->d_name);
          // build the overall filename
          strcpy(buffer, dirname);
          strcat(buffer, "/");
          strcat(buffer, dp->d_name);
          
          cv::Mat img = imread(buffer, cv::IMREAD_COLOR);
          std::vector<float> imageData;
          
          // Reset/Erase a file if it was the first iteration
          int reset = (iter == 0) ? 1 : 0;
          switch (featureType) {
              case 1:{
                  // Feature = the middle 9x9 pixels
                  extractMiddleVector(img, 9, 9, imageData);
                  append_image_data_csv(MIDDLE_FEATURE, buffer, imageData, reset);
                  break;
              }
              case 2:{
                  // Feature = 3D Histogram with bins of 8 each
                  int bins = 8;
                  extract3DHistVector(img, bins, imageData);
                  append_image_data_csv(HIST_FEATURE, buffer, imageData, reset);
                  break;
              }
              case 3:{
                  // Split image to top and bottom
                  // 3D Histogram with bins of 8 each
                  cv::Rect upperHalf(0, 0, img.cols-1, (img.rows-1)/2);
                  cv::Rect lowerHalf(0, img.rows/2+1, img.cols-1, (img.rows-1)/2);
                  
                  cv::Mat upperImg = img(upperHalf);
                  cv::Mat lowerImg = img(lowerHalf);
                  
                  // Feature = 3D Histogram with bins of 8 each
                  int bins = 8;
                  extract3DHistVector(upperImg, bins, imageData);
                  append_image_data_csv(HIST_UPPERHALF_FEATURE, buffer, imageData, reset);
                  
                  std::vector<float> imageDataTwo;
                  extract3DHistVector(lowerImg, bins, imageDataTwo);
                  append_image_data_csv(HIST_LOWERHALF_FEATURE, buffer, imageDataTwo, reset);
                  break;
              }
              case 4: {
                  // 3D Histogram with bins of 8 each +
                  // 3D Histogram of Sobel Magnitude with bins of 8 each
                  // 3D Histogram of Gobar Filter with bins of 8 each
                  int bins = 8;
                  extract3DHistVector(img, bins, imageData);
                  append_image_data_csv(HIST_FEATURE, buffer, imageData, reset);
                  
                  std::vector<float> imageDataTwo;
                  extractSobelTextureVector(img, bins, imageDataTwo);
                  append_image_data_csv(HIST_SOBEL_TEXTURE_FEATURE, buffer, imageDataTwo, reset);
                  break;
              }
              case 5:{
                  // Only use the middle 100x100 and 50x50 pixels
                  // 3D Histogram with bins of 8 each
                  // 3D Histogram of Gobar Filter with bins of 8 each
                  int bins = 8;
                  
                  int midRow = (img.rows%2 == 0)? img.rows/2 : img.rows/2+1;
                  int midCol = (img.cols%2 == 0)? img.cols/2 : img.cols/2+1;
                  int sizeMid = 100;
                  int sizeSmall = 50;
                  
                  cv::Rect middle(midCol-sizeMid/2, midRow-sizeMid/2, sizeMid, sizeMid);
                  cv::Rect smaller(midCol-sizeSmall/2, midRow-sizeSmall/2, sizeSmall, sizeSmall);
                  
                  cv::Mat middleImg = img(middle);
                  cv::Mat smallerImg = img(smaller);
                
                  extract3DHistVector(middleImg, bins, imageData);
                  append_image_data_csv(HIST_MIDDLE_MED_FEATURE, buffer, imageData, reset);
                  
                  std::vector<float> imageDataTwo, imageDataThree, imageDataFour;
                  extractGaborTextureVector(middleImg, bins, imageDataTwo);
                  append_image_data_csv(HIST_MIDDLE_MED_GABOR_FEATURE, buffer, imageDataTwo, reset);
                  
                  extract3DHistVector(smallerImg, bins, imageDataThree);
                  append_image_data_csv(HIST_MIDDLE_SMALL_FEATURE, buffer, imageDataThree, reset);
                  
                  extractGaborTextureVector(smallerImg, bins, imageDataFour);
                  append_image_data_csv(HIST_MIDDLE_SMALL_GABOR_FEATURE, buffer, imageDataFour, reset);
                  break;
              }
              case 6:{
                  // 3D SOFT Histogram with bins of 8 each, and softWidth of 5
                  int bins = 8;
                  int softWidth = 5;
                  
                  extract3DSoftHistVector(img, bins, softWidth, imageData);
                  append_image_data_csv(HIST_SOFT_FEATURE, buffer, imageData, reset);
                  break;
              }
              case 7:{
                  // 3D Histogram with bins of 8 each +
                  // 3D Histogram on Law's Filter Averaged, with bins of 8 each
                  
                  int bins = 8;
                  extract3DHistVector(img, bins, imageData);
                  append_image_data_csv(HIST_FEATURE, buffer, imageData, reset);
                  
                  std::vector<float> imageDataTwo;
                  extractLawsTextureVector(img, bins, imageDataTwo);
                  append_image_data_csv(HIST_LAWS_FEATURE, buffer, imageDataTwo, reset);
                  break;
              }
              case 8: {
                  // 3D Histogram on Sobel Magnitude, with bins of 8 each
                  int bins = 8;
                  std::vector<float> imageDataTwo;
                  extractSobelTextureVector(img, bins, imageData);
                  append_image_data_csv(HIST_SOBEL_TEXTURE_FEATURE, buffer, imageData, reset);
                  break;
              }
              case 9:{
                  //3D Histogram on Law's Filter Averaged, with bins of 8 each
                  int bins = 8;
                  extractLawsTextureVector(img, bins, imageData);
                  append_image_data_csv(HIST_LAWS_FEATURE, buffer, imageData, reset);
                  break;
              }
              case 10:{
                  // 3D Histogram on Gabor's Filter, with bins of 8 each
                  int bins = 8;
                  extractGaborTextureVector(img, bins, imageData);
                  append_image_data_csv(HIST_GABOR_FEATURE, buffer, imageData, reset);
                  break;
              }
              default:{
                  printf("Incorrect featureType input number");
                  exit(-1);
                  break;
              }
          }
          iter+=1;
      }
    }
    
    return 0;
}

// Find the K most similar images (paths) given a target image.
// targetImg - Target Image to be matched to
// featureType - Feature Type, ranging from 1 - 10
// matchingMethod - matching method, ranging from 1 - 2. aka distance metric
// k - Number of top matching images to be returned
// topKFileNames - FileNames of the top K matching images
int knn(cv::Mat &targetImg,
        int featureType,
        int matchingMethod,
        int k,
        std::vector<char *> &topKFileNames
        ){
    std::vector<std::vector<float>> imageDataVec(50, std::vector<float>());
    std::vector<char *> csvVec;
    std::vector<double> weightVec;
    float(*distanceMetric)(std::vector<float>, std::vector<float>);
    
    switch(matchingMethod) {
        case 1:{
            distanceMetric = &sumSquared;
            break;
        }
        case 2: {
            distanceMetric = &histIntersectionNormalized;
            break;
        }
        default:{
            printf("Incorrect matchingMethod input number");
            exit(-1);
        }
    }
    
    switch (featureType) {
        case 1:{
            // The middle 9x9 pixels
            distanceMetric = &sumSquared;
            csvVec.push_back(MIDDLE_FEATURE);
            weightVec.push_back(1.0);
            
            extractMiddleVector(targetImg, 9, 9, imageDataVec[0]);
            break;
        }
        case 2:{
            // 3D Histogram with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_FEATURE);
            weightVec.push_back(1.0);
            
            extract3DHistVector(targetImg, bins, imageDataVec[0]);
            break;
        }
        case 3:{
            // Split image to top and bottom
            // 3D Histogram with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_UPPERHALF_FEATURE);
            csvVec.push_back(HIST_LOWERHALF_FEATURE);
            weightVec.push_back(0.5);
            weightVec.push_back(0.5);
            
            cv::Rect upperHalf(0, 0, targetImg.cols-1, (targetImg.rows-1)/2);
            cv::Rect lowerHalf(0, targetImg.rows/2+1, targetImg.cols-1, (targetImg.rows-1)/2);
            cv::Mat upperImg = targetImg(upperHalf);
            cv::Mat lowerImg = targetImg(lowerHalf);
            
            extract3DHistVector(upperImg, bins, imageDataVec[0]);
            extract3DHistVector(lowerImg, bins, imageDataVec[1]);
            break;
        }
        case 4:{
            // 3D Histogram with bins of 8 each +
            // 3D Histogram of Sobel Magnitude with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_FEATURE);
            csvVec.push_back(HIST_SOBEL_TEXTURE_FEATURE);
            weightVec.push_back(0.5);
            weightVec.push_back(0.5);
            
            extract3DHistVector(targetImg, bins, imageDataVec[0]);
            extractSobelTextureVector(targetImg, bins, imageDataVec[1]);
            break;
        }
        case 5:{
            // Only use the middle 100x100 and 50x50 pixels
            // 3D Histogram with bins of 8 each
            // 3D Histogram of Gobar Filter with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_MIDDLE_MED_FEATURE);
            csvVec.push_back(HIST_MIDDLE_MED_GABOR_FEATURE);
            csvVec.push_back(HIST_MIDDLE_SMALL_FEATURE);
            csvVec.push_back(HIST_MIDDLE_SMALL_GABOR_FEATURE);
            weightVec.push_back(0.1);
            weightVec.push_back(0.05);
            weightVec.push_back(0.65);
            weightVec.push_back(0.2);
            
            int midRow = (targetImg.rows%2 == 0)? targetImg.rows/2 : targetImg.rows/2+1;
            int midCol = (targetImg.cols%2 == 0)? targetImg.cols/2 : targetImg.cols/2+1;
            int sizeMid = 100;
            int sizeSmall = 50;
            
            cv::Rect middle(midCol-sizeMid/2, midRow-sizeMid/2, sizeMid, sizeMid);
            cv::Rect smaller(midCol-sizeSmall/2, midRow-sizeSmall/2, sizeSmall, sizeSmall);
            
            cv::Mat middleImg = targetImg(middle);
            cv::Mat smallerImg = targetImg(smaller);
            
            extract3DHistVector(middleImg, bins, imageDataVec[0]);
            extractGaborTextureVector(middleImg, bins, imageDataVec[1]);
            extract3DHistVector(smallerImg, bins, imageDataVec[2]);
            extractGaborTextureVector(smallerImg, bins, imageDataVec[3]);
            break;
        }
        case 6: {
            // 3D SOFT Histogram with bins of 8 each, and softWidth of 5
            int bins = 8;
            int softWidth = 5;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_SOFT_FEATURE);
            weightVec.push_back(1.0);

            extract3DSoftHistVector(targetImg, bins, softWidth, imageDataVec[0]);
            break;
        }
        case 7:{
            // 3D Histogram with bins of 8 each +
            // 3D Histogram on Law's Filter Averaged, with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_FEATURE);
            csvVec.push_back(HIST_LAWS_FEATURE);
            weightVec.push_back(0.5);
            weightVec.push_back(0.5);
            
            extract3DHistVector(targetImg, bins, imageDataVec[0]);
            extractLawsTextureVector(targetImg, bins, imageDataVec[1]);
            break;
        }
        case 8:{
            // 3D Histogram of Sobel Magnitude with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_SOBEL_TEXTURE_FEATURE);
            weightVec.push_back(1.0);
            
            extractSobelTextureVector(targetImg, bins, imageDataVec[0]);
            break;
        }
        case 9:{
            // 3D Histogram on Law's Filter Averaged, with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_LAWS_FEATURE);
            weightVec.push_back(1.0);
            
            extractLawsTextureVector(targetImg, bins, imageDataVec[0]);
            break;
        }
        case 10:{
            // 3D Histogram of Gabor's Filter with bins of 8 each
            int bins = 8;
            distanceMetric = &histIntersectionNormalized;
            csvVec.push_back(HIST_GABOR_FEATURE);
            weightVec.push_back(1.0);
            
            extractGaborTextureVector(targetImg, bins, imageDataVec[0]);
            break;
        }
        default:{
            printf("Incorrect featureType input number");
            exit(-1);
        }
    }
    
    // Compute distance
    // Loop through csvVec with has the same size of (populated) imageDataVec
    std::vector<char *> filenames;
    std::vector<std::pair<float, char *>> distances;
    for (int i = 0; i<csvVec.size(); i++){
        std::vector<std::vector<float>> precomputeData;
        std::vector<char *> filenames;
        read_image_data_csv(csvVec[i], filenames, precomputeData,0);
        
        // For each csv/imageData, Loop and compare to precompute Data
        for(int j = 0; j < precomputeData.size(); j++) {
            float distance = weightVec[i] * distanceMetric(imageDataVec[i], precomputeData[j]);
            if (i==0){
                std::pair<float, char *> distPair(distance, filenames[j]);
                distances.push_back(distPair);
            } else {
                distances[j].first += distance;
            }
        }
    }

    // Look for the top K (smallest distance)
    std::sort(distances.begin(), distances.end());
    for (int i =0; i<k; i++){
        topKFileNames.push_back(distances[i].second);
    }
    
    return 0;
}

int main(int argc, char *argv[]) {
    /*
     argv[0] - cpp filename
     argv[1] - target filename for T
     argv[2] - directory of images as the database B
     argv[3] - feature type, ranging from 1 - 10
     argv[4] - matching method, ranging from 1 - 2
     argv[5] - the number of images N to return
     argv[6] - compute feature vector for each image in database B. Set this to zero if doesn't want to compute feature vector
     */

    char targetImgPath[256];
    char imgDir[256];
    int featureType;
    int matchingMethod; // aka distanceMetric
    int N;
    int createFeatureVecs;
    
    //Parse argv
    strcpy(targetImgPath, argv[1]);
    strcpy(imgDir, argv[2]);
    featureType = atoi(argv[3]);
    matchingMethod = atoi(argv[4]);
    N = atoi(argv[5]);
    createFeatureVecs = atoi(argv[6]);
    
    // Find the top K matching images
    std::vector<char *> topNFileNames;
    cv::Mat img = imread(targetImgPath, cv::IMREAD_COLOR);
    
    if (createFeatureVecs) createFeatureVector(imgDir, featureType);
    knn(img, featureType, matchingMethod, N+1, topNFileNames);
    std::vector<cv::Mat> topNFileMatrices;
    for (char *topFn : topNFileNames) {
        topNFileMatrices.push_back(cv::imread(topFn));
        std::cout<<topFn<<std::endl;
    }
    
    // Showing the input and top K matching images in a row
    cv::Mat imgShow;
    cv::hconcat(topNFileMatrices, imgShow);
    for(;;){
        cv::imshow("Top K Matching Images", imgShow);
        int key = cv::waitKey(0); // 0 means wait forever
        if (key =='q') break;
        if (key == 's'){
            std::string str = argv[1];
            std::string filteredFn;
            for (char c : str) {
                if (isdigit(c)) {
                    filteredFn += c;
                }
            }
            filteredFn+="_featureType";
            filteredFn+=argv[3];
            filteredFn+="_matchingMethod";
            filteredFn+=argv[4];
            filteredFn+=".png";
            cv::imwrite(filteredFn, imgShow);
            std::cout<<filteredFn<<std::endl;
        }
    }
    return 0;
}
