package com.knock.ridereport.sensor.RandomForest;

import com.sun.jna.Native;
import com.sun.jna.Structure;
import com.sun.jna.Pointer;
import com.sun.jna.PointerType;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.lang.UnsatisfiedLinkError;

import timber.log.Timber;

/**
 * Created by evan on 4/8/16.
 */
public class RandomForestAdapterJNA implements RandomForestAdapter {
    public static class RFManagerPtr extends PointerType {}

    private static native RFManagerPtr createRandomForestManager(int sampleCount, int samplingRateHz, String pathToModelFile);
    private static native void randomForestClassifyMagnitudeVector(RFManagerPtr manager, float[] accNorms, float[] confidences, int n_classes);
    private static native int randomForestGetClassCount(RFManagerPtr manager);
    private static native void randomForestGetClassLabels(RFManagerPtr manager, int[] labels, int labelCount);
    private static native boolean randomForestClassifyAccelerometerSignal(RFManagerPtr manager, AccelerometerReading.ByReference signal, int readingCount, float[] confidences, int n_classes);

    static {
		try {
		    System.loadLibrary("jnidispatch");
		}
		catch (UnsatisfiedLinkError e) {
		    if (System.getProperty("java.vm.name").equalsIgnoreCase("Dalvik")) {
		        Timber.d("Failed to load jnidispatch; trying to load Native anyway.");
		    }
		    else {
		        Timber.d("jnidispatch did not load; this is usually OK off Android");
		    }
		}
        Native.register(RandomForestAdapterJNA.class, "rrnative");
    }

    private RFManagerPtr _manager;
    private int _classCount;
    private int[] _classLabels;

    public RandomForestAdapterJNA(int sampleCount, int samplingRateHz, String filename) throws FileNotFoundException {
        _manager = createRandomForestManager(sampleCount, samplingRateHz, filename);
        Timber.d("Getting class count");
        _classCount = randomForestGetClassCount(_manager);

        _classLabels = new int[_classCount];
        Timber.d("Getting class labels");
        randomForestGetClassLabels(_manager, _classLabels, _classCount);
    }

    public float[] classifyAccelerometerSignal(List<? extends SensorDataInterface> sensorDataList) throws IllegalArgumentException {
        if (sensorDataList.size() == 0) {
            throw new IllegalArgumentException("Cannot classify empty list of sensor readings");
        }
        // Structure.toArray() allocates a contiguous memory block for the structs
        final AccelerometerReading[] readings = (AccelerometerReading[]) (new AccelerometerReading.ByReference()).toArray(sensorDataList.size());

        for (int i = 0; i < readings.length; ++i) {
            SensorDataInterface sensorData = sensorDataList.get(i);
            readings[i].x = sensorData.getX();
            readings[i].y = sensorData.getY();
            readings[i].z = sensorData.getZ();
            readings[i].t = sensorData.getSeconds();
            // System.err.println("java reading " + readings[i].x + " " + readings[i].y + " " + readings[i].z + " " + readings[i].t);
        }

        float[] confidences = new float[_classCount];

        boolean successful = randomForestClassifyAccelerometerSignal(_manager, (AccelerometerReading.ByReference)readings[0], readings.length, confidences, _classCount);

        if (!successful) {
            Timber.d("classifyAccelerometerSignal failed");
        }
        Timber.d("classifyAccelerometerSignal labels: " + Arrays.toString(_classLabels));
        Timber.d("classifyAccelerometerSignal confidences: " + Arrays.toString(confidences));
        return confidences;
    }

    public int[] getClassLabels() {
        return _classLabels;
    }

}
