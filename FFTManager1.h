typedef struct FFTManager1 FFTManager1;

FFTManager1* createFFTManager1(int sampleSize);
void fft(float * input, int inputSize, float *output, FFTManager1 *manager);
void deleteFFTManager1(FFTManager1 *manager);
float dominantPower1(float *input, int inputSize);
