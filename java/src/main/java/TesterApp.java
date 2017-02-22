package com.knock.ridereport.sensor.RandomForest;

import com.knock.ridereport.sensor.RandomForest.RandomForestAdapterJNA;
import java.io.FileNotFoundException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

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
            System.out.println(forestPath);
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

        if (method.equals("predictConfidences")) {
            try {
                JSONObject output = doPredictConfidences(adapter, obj);
                return output.toString();
            }
            catch (JSONException e) {
                return errorObject(e.toString()).toString();
            }
            catch (Exception e) {
                return errorObject(e.toString()).toString();
            }
        }
        else {
            return errorObject("Unknown method '" + method + "'").toString();
        }
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
}
