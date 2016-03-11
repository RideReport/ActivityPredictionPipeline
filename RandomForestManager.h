//
//  RandomForestManager.h
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#define RANDOM_FOREST_VECTOR_SIZE (7)

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct RandomForestManager RandomForestManager;
    RandomForestManager *createRandomForestManager(int sampleSize, const char* pathToModelFile);
    void prepFeatureVector(RandomForestManager *randomForestManager, float *features, float *magnitudes);
    void deleteRandomForestManager(RandomForestManager *r);
    int randomForesetClassifyMagnitudeVector(RandomForestManager *randomForestManager, float *magnitudeVector);
    void randomForestClassificationConfidences(RandomForestManager *randomForestManager, float *magnitudeVector, float *confidences, int n_classes);
    int randomForestGetClassCount(RandomForestManager *randomForestManager);
    int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int n_classes);
#ifdef __cplusplus
}
#endif
