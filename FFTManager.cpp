#include "FFTManager.h"

struct FFTManager {
};

FFTManager* createFFTManager(int sampleSize) {
  return new FFTManager;
}

void fft(float * input, int inputSize, float *output, FFTManager *manager) {
    
}

float dominantPower(float *input, int inputSize) {
    return 0.0;
}
