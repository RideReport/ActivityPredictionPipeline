#include <stdlib.h>
#include "FFTManager.h"

struct FFTManager {
    float thing = 0.0;
};

FFTManager* createFFTManager(int sampleSize) {
  return (struct FFTManager*) malloc(sizeof(struct FFTManager));
}

void fft(float * input, int inputSize, float *output, FFTManager *manager) {
    
}

float dominantPower(float *input, int inputSize) {
    return 0.0;
}
