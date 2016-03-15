//
//  RandomForestManager.h
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#define RANDOM_FOREST_VECTOR_SIZE (8)
#ifdef __cplusplus
extern "C" {
#endif
    typedef struct RandomForestManager RandomForestManager;
    RandomForestManager *createRandomForestManager(int sampleSize, const char* pathToModelFile);
    void deleteRandomForestManager(RandomForestManager *r);
	float dominantPowerOfFFT(RandomForestManager *randomForestManager, float * input, int inputSize, int managerType);
    void prepFeatureVector(RandomForestManager *randomForestManager, float *features, float *magnitudes, float *speedVector, int speedVectorCount);
    int randomForesetClassifyMagnitudeVector(RandomForestManager *randomForestManager, float *magnitudeVector, float *speedVector, int speedVectorCount);
    void randomForestClassificationConfidences(RandomForestManager *randomForestManager, float *magnitudeVector, float *speedVector, int speedVectorCount, float *confidences, int classCount);
    int randomForestGetClassCount(RandomForestManager *randomForestManager);
    int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int classCount);
#ifdef __cplusplus
}
#endif
