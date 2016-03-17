#include <stdlib.h>
#include "FFTManager_fftw.h"
#include <fftw3.h>
#include <math.h>


struct FFTManager {
    unsigned int N;
    fftw_complex *in;
    fftw_complex *out;
    float* multipliers;
    fftw_plan p;
};

void setupHammingWindow(float *values, int N) {
    for (int i = 0; i < N; ++i) {
        values[i] = 0.54 - 0.46 * cos(2*M_PI*i/(N-1));
    }
}

FFTManager* createFFTManager(int sampleSize) {
  FFTManager* _fft = (struct FFTManager*) malloc(sizeof(struct FFTManager));
  _fft->N = sampleSize;
  _fft->in = fftw_alloc_complex(sampleSize);
  _fft->out = fftw_alloc_complex(sampleSize);
  _fft->p = fftw_plan_dft_1d(sampleSize, _fft->in, _fft->out, FFTW_FORWARD, FFTW_MEASURE);

  _fft->multipliers = (float*) malloc(sizeof(float) * sampleSize);
  setupHammingWindow(_fft->multipliers, sampleSize);

  return _fft;
}

void fft(FFTManager *_fft, float * input, int inputSize, float *output) {
    if (inputSize != _fft->N) {
        // throw?
        return;
    }


    for (int i = 0; i < inputSize; ++i) {
        _fft->in[i][0] = input[i] * _fft->multipliers[i];
        _fft->in[i][1] = 0.0;
    }

    fftw_execute(_fft->p);

    // Compute *squared* magnitudes
    for (int i = 0; i <= _fft->N/2; ++i) {
        output[i] = (_fft->out[i][0] * _fft->out[i][0]) + (_fft->out[i][1] * _fft->out[i][1]);
    }
}

void deleteFFTManager(FFTManager *_fft) {
    fftw_destroy_plan(_fft->p);
    fftw_free(_fft->in);
    fftw_free(_fft->out);
    free(_fft);
}

float dominantPower(float *output, int inputSize) {
    float max = 0.0;
    for (int i = 1; i <= inputSize/2; ++i) {
        if (output[i] > max) {
            max = output[i];
        }
    }

    return max;
}
