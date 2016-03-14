typedef struct FFTManager FFTManager;

FFTManager* createFFTManager(int sampleSize);
void fft(float * input, int inputSize, float *output, FFTManager *manager);
void deleteFFTManager(FFTManager *manager);
float dominantPower(float *input, int inputSize);
