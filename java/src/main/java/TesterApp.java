package com.knock.ridereport.sensor.RandomForest;

import com.knock.ridereport.sensor.RandomForest.RandomForestAdapterJNA;
import java.io.FileNotFoundException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.PrintWriter;

import java.util.ArrayList;

import org.json.*;


public class TesterApp {
    public static void main(String[] args) throws FileNotFoundException {
        String existingPath = System.getProperty("jna.library.path");
        System.setProperty("jna.library.path",
            existingPath
            + ":" + System.getProperty("user.dir")
            + ":" + System.getProperty("user.dir") + "/mode_classification_wrapper/java");

        String forestPath = "data/forestAccelOnly.cv";
        if (args.length >= 1) {
            forestPath = args[0];
        }
        RandomForestAdapterJNA adapter = new RandomForestAdapterJNA(64, 20, forestPath);

        System.err.println("Reading JSON from stdin");

        try {
            BufferedReader inputReader = new BufferedReader(new InputStreamReader(System.in));
            System.out.println(readyObject(adapter).toString());
            do {
                String line = inputReader.readLine();
                if (line == null) {
                    break;
                }

                System.out.println(dispatch(adapter, line));
            }
            while (true);
        } catch (IOException e) {
            // pass
        }

    }

    public static String dispatch(RandomForestAdapterJNA adapter, String line) {
        JSONObject obj = new JSONObject(line);
        String method;
        try {
            method = obj.getString("method");
        }
        catch (JSONException e) {
            return errorObject(e.toString()).toString();
        }

        try {
            if (method.equals("predictConfidences")) {
                return doPredictConfidences(adapter, obj).toString();
            }
            else if (method.equals("classifyAccelerometerSignal")) {
                return doClassifyAccelerometerSignal(adapter, obj).toString();
            }
            else {
                return errorObject("Unknown method '" + method + "'").toString();
            }
        }
        catch (Exception e) {
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            return errorObject(sw.toString()).toString();
        }
    }

    public static JSONObject doClassifyAccelerometerSignal(RandomForestAdapterJNA adapter, JSONObject obj) throws JSONException, IllegalArgumentException {
        JSONArray readingsJson = obj.getJSONArray("readings");

        ArrayList<RandomForestAdapterJNA.SensorDataInterface> sensorDataList = new ArrayList<>(readingsJson.length());

        for (int i = 0; i < readingsJson.length(); ++i) {
            JSONObject reading = readingsJson.getJSONObject(i);
            MockSensorData sensorData = new MockSensorData();

            sensorData.x = (float) reading.getDouble("x");
            sensorData.y = (float) reading.getDouble("y");
            sensorData.z = (float) reading.getDouble("z");
            sensorData.t = (float) reading.getDouble("t");
            sensorDataList.add(i, sensorData);
        }
        float[] confidences = adapter.classifyAccelerometerSignal(sensorDataList);
        int[] labels = adapter.getClassLabels();

        JSONObject output = new JSONObject();
        for (int i = 0; i < labels.length; ++i) {
            output.put(Integer.toString(labels[i]), confidences[i]);
        }
        return output;
    }

    public static JSONObject doPredictConfidences(RandomForestAdapterJNA adapter, JSONObject obj) throws JSONException, IllegalArgumentException {
        JSONArray accNormsJson = obj.getJSONArray("accNorms");
        float[] accNorms = new float[accNormsJson.length()];
        for (int i = 0; i < accNorms.length; ++i) {
            accNorms[i] = (float) accNormsJson.getDouble(i);
        }
        float[] confidences = adapter.predictConfidences(accNorms);
        int[] labels = adapter.getClassLabels();

        JSONObject output = new JSONObject();
        for (int i = 0; i < labels.length; ++i) {
            output.put(Integer.toString(labels[i]), confidences[i]);
        }
        return output;
    }

    public static JSONObject errorObject(String detail) {
        JSONObject output = new JSONObject();
        output.put("detail", detail);
        output.put("error", true);
        return output;
    }

    public static JSONObject readyObject(RandomForestAdapterJNA adapter) {
        JSONObject output = new JSONObject();
        output.put("ready", true);
        return output;
    }

    static class MockSensorData implements RandomForestAdapterJNA.SensorDataInterface {
        public float x;
        public float y;
        public float z;
        public float t;

        public float getX() { return x; }
        public float getY() { return y; }
        public float getZ() { return z; }
        public float getFloatSeconds() { return t; }
    }
}
