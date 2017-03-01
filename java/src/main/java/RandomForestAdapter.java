package com.knock.ridereport.sensor.RandomForest;

import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * Created by evan on 6/15/16.
 */
public abstract interface RandomForestAdapter {
    String TAG = "RandomForestAdapter";
    int SAMPLE_SIZE = 64;
    int SAMPLING_RATE_HZ = 20;

    public interface SensorDataInterface {
        public float getX();
        public float getY();
        public float getZ();
        public double getSeconds();
    };

    public static class AccelerometerReading extends Structure {
        public static class ByReference extends AccelerometerReading implements Structure.ByReference {}

        public float x;
        public float y;
        public float z;
        public double t;

        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList(new String[] { "x", "y", "z", "t" });
        }
    }

    abstract public float[] classifyAccelerometerSignal(List<? extends SensorDataInterface> sensorDataList);
    abstract public int[] getClassLabels();
}
