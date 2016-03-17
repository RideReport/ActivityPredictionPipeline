typedef struct FFTManager FFTManager;

FFTManager* createFFTManager(int sampleSize);
void fft(FFTManager *manager, float * input, int inputSize, float *output);
void deleteFFTManager(FFTManager *manager);
float dominantPower(float *input, int inputSize);
