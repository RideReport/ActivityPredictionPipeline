package com.knock.ridereport.sensor.RandomForest;

/**
 * Created by evan on 6/15/16.
 */
public abstract interface RandomForestAdapter {
    String TAG = "RandomForestAdapter";
    int SAMPLE_SIZE = 64;
    int SAMPLING_RATE_HZ = 20;

    abstract public float[] predictBestTypeAndConfidence(float[] accNorms, float[] gyroNorms);
    abstract public float[] prepareAccelerometerOnlyFeatures(float[] accNorms);
    abstract Boolean isAcclereomterOnlyVersion();
}
