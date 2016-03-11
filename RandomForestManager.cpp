//
//  RandomForestManager
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#include "RandomForestManager.h"
#include "FFTManager.h"
#include<stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

using namespace cv;

// Private Functions
float max(cv::Mat mat);
double maxMean(cv::Mat mat, int windowSize);
double skewness(cv::Mat mat);
double kurtosis(cv::Mat mat);

struct RandomForestManager {
    int sampleSize;
    FFTManager *fftManager;
    cv::Ptr<cv::ml::RTrees> model;
};

RandomForestManager *createRandomForestManager(int sampleSize, const char* pathToModelFile)
{
    assert(fmod(log2(sampleSize), 1.0) == 0.0); // sampleSize must be a power of 2
    
    RandomForestManager *r = new RandomForestManager;
    r->sampleSize = sampleSize;
    r->fftManager = createFFTManager(sampleSize);
    r->model = cv::ml::RTrees::load<cv::ml::RTrees>(pathToModelFile);

    return r;
}

void deleteRandomForestManager(RandomForestManager *r)
{
    free(r->fftManager);
    delete r->model;
    free(r);
}

void prepFeatureVector(RandomForestManager *randomForestManager, float* features, float* mags_raw) {
    cv::Scalar mean,stddev;
    cv::Mat mags = cv::Mat(randomForestManager->sampleSize, 1, CV_32F, mags_raw);
    meanStdDev(mags,mean,stddev);
    
    float *fftOutput = new float[randomForestManager->sampleSize];
    fft(mags_raw, randomForestManager->sampleSize, fftOutput, randomForestManager->fftManager);
    float maxPower = dominantPower(fftOutput, randomForestManager->sampleSize);

    features[0] = max(mags);
    features[1] = (float)mean.val[0];
    features[2] = maxMean(mags, 5);
    features[3] = (float)stddev.val[0];
    features[4] = (float)skewness(mags);
    features[5] = (float)kurtosis(mags);
    features[6] = maxPower;
}

int randomForesetClassifyMagnitudeVector(RandomForestManager *randomForestManager, float *magnitudeVector)
{
    cv::Mat readings = cv::Mat::zeros(1, RANDOM_FOREST_VECTOR_SIZE, CV_32F);

    prepFeatureVector(randomForestManager, readings.ptr<float>(), magnitudeVector);
    
    return (int)randomForestManager->model->predict(readings, cv::noArray(), cv::ml::DTrees::PREDICT_MAX_VOTE);
}

#include <iostream>
using namespace std;

void randomForestClassificationConfidences(RandomForestManager *randomForestManager, float *magnitudeVector, float *confidences, int n_classes) {
    cv::Mat readings = cv::Mat::zeros(1, RANDOM_FOREST_VECTOR_SIZE, CV_32F);

    prepFeatureVector(randomForestManager, readings.ptr<float>(), magnitudeVector);

    cout << "readings " <<  readings << endl;
    cout << "at 0,5 " << readings.at<float>(0, 5) << endl;
    cout << "at 0,6 " << readings.at<float>(0, 6) << endl;

    cv::Mat results;

    randomForestManager->model->predictProb(readings, results, cv::ml::DTrees::PREDICT_CONFIDENCE);

    for (int i = 0; i < n_classes; ++i) {
        confidences[i] = results.at<float>(i);
    }
}

int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int n_classes) {
    Mat labelsMat = randomForestManager->model->getClassLabels();
    for (int i = 0; i < n_classes && i < labelsMat.rows; ++i) {
        labels[i] = labelsMat.at<int>(i);
    }
    return labelsMat.rows;
}

int randomForestGetClassCount(RandomForestManager *randomForestManager) {
    Mat labelsMat = randomForestManager->model->getClassLabels();
    return labelsMat.rows;
}

float max(cv::Mat mat)
{
    float max = 0;
    for (int i=0;i<mat.rows;i++)
    {
        float elem = mat.at<float>(i,0);
        if (elem > max) {
            max = elem;
        }
    }
    
    return max;
}

double maxMean(cv::Mat mat, int windowSize)
{
    if (windowSize>mat.rows) {
        return 0;
    }
    
    cv::Mat rollingMeans = cv::Mat::zeros(mat.rows - windowSize, 1, CV_32F);
    
    for (int i=0;i<=(mat.rows - windowSize);i++)
    {
        float sum = 0;
        for (int j=0;j<windowSize;j++) {
            sum += mat.at<float>(i+j,0);
        }
        rollingMeans.at<float>(i,0) = sum/windowSize;
    }
    
    double min, max;
    cv::minMaxLoc(rollingMeans, &min, &max);
    
    return max;
}

double skewness(cv::Mat mat)
{
    cv::Scalar skewness,mean,stddev;
    skewness.val[0]=0;
    skewness.val[1]=0;
    skewness.val[2]=0;
    meanStdDev(mat,mean,stddev,cv::Mat());
    int sum0, sum1, sum2;
    float den0=0,den1=0,den2=0;
    int N=mat.rows*mat.cols;
    
    for (int i=0;i<mat.rows;i++)
    {
        for (int j=0;j<mat.cols;j++)
        {
            sum0=mat.ptr<uchar>(i)[3*j]-mean.val[0];
            sum1=mat.ptr<uchar>(i)[3*j+1]-mean.val[1];
            sum2=mat.ptr<uchar>(i)[3*j+2]-mean.val[2];
            
            skewness.val[0]+=sum0*sum0*sum0;
            skewness.val[1]+=sum1*sum1*sum1;
            skewness.val[2]+=sum2*sum2*sum2;
            den0+=sum0*sum0;
            den1+=sum1*sum1;
            den2+=sum2*sum2;
        }
    }
    
    skewness.val[0]=skewness.val[0]*sqrt(N)/(den0*sqrt(den0));
    skewness.val[1]=skewness.val[1]*sqrt(N)/(den1*sqrt(den1));
    skewness.val[2]=skewness.val[2]*sqrt(N)/(den2*sqrt(den2));
    
    return skewness.val[0];
}

double kurtosis(cv::Mat mat)
{
    cv::Scalar kurt,mean,stddev;
    kurt.val[0]=0;
    kurt.val[1]=0;
    kurt.val[2]=0;
    meanStdDev(mat,mean,stddev,cv::Mat());
    int sum0, sum1, sum2;
    int N=mat.rows*mat.cols;
    float den0=0,den1=0,den2=0;
    
    for (int i=0;i<mat.rows;i++)
    {
        for (int j=0;j<mat.cols;j++)
        {
            sum0=mat.ptr<uchar>(i)[3*j]-mean.val[0];
            sum1=mat.ptr<uchar>(i)[3*j+1]-mean.val[1];
            sum2=mat.ptr<uchar>(i)[3*j+2]-mean.val[2];
            
            kurt.val[0]+=sum0*sum0*sum0*sum0;
            kurt.val[1]+=sum1*sum1*sum1*sum1;
            kurt.val[2]+=sum2*sum2*sum2*sum2;
            den0+=sum0*sum0;
            den1+=sum1*sum1;
            den2+=sum2*sum2;
        }
    }
    
    kurt.val[0]= (kurt.val[0]*N*(N+1)*(N-1)/(den0*den0*(N-2)*(N-3)))-(3*(N-1)*(N-1)/((N-2)*(N-3)));
    kurt.val[1]= (kurt.val[1]*N/(den1*den1))-3;
    kurt.val[2]= (kurt.val[2]*N/(den2*den2))-3;
    
    return kurt.val[0];
}
