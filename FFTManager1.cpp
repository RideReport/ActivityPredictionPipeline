#include <stdlib.h>
#include "FFTManager1.h"
#include <fftw3.h>
#include <math.h>


struct FFTManager1 {
    unsigned int N;
    fftw_complex *in;
    fftw_complex *out;
    double* outMagsSquared;
    fftw_plan p;
};

FFTManager1* createFFTManager1(int sampleSize) {
  FFTManager1* _fft = (struct FFTManager1*) malloc(sizeof(struct FFTManager1));
  _fft->N = sampleSize;
  _fft->in = fftw_alloc_complex(sampleSize);
  _fft->out = fftw_alloc_complex(sampleSize);
  _fft->p = fftw_plan_dft_1d(sampleSize, _fft->in, _fft->out, FFTW_FORWARD, FFTW_MEASURE);

  return _fft;
}

void fft1(float * input, int inputSize, float *output, FFTManager1 *_fft) {
    if (inputSize != _fft->N) {
        // throw?
        return;
    }

    for (int i = 0; i < inputSize; ++i) {
        _fft->in[i][0] = input[i];
        _fft->in[i][1] = 0.0;
    }

    fftw_execute(_fft->p);

    // Compute *squared* magnitudes
    for (int i = 0; i < _fft->N/2; ++i) {
        output[i] = (_fft->out[i][0] * _fft->out[i][0]) + (_fft->out[i][1] * _fft->out[i][1]);
    }
}

void deleteFFTManager1(FFTManager1 *_fft) {
    fftw_destroy_plan(_fft->p);
    fftw_free(_fft->in);
    fftw_free(_fft->out);
    free(_fft);
}

float dominantPower1(float *output, int inputSize) {
    float max = 0.0;
    for (int i = 0; i < inputSize/2; ++i) {
        if (output[i] > max) {
            max = output[i];
        }
    }

    return sqrt(max);
}