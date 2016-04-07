import com.sun.jna.Native;
import com.sun.jna.PointerType;


public class RandomForestAdapter {
  protected static class RFManagerPtr extends jna.PointerType {}
  protected static native createRandomForestManager(int sampleCount, int samplingRateHz, String pathToModelFile);
  protected static native randomForestClassificationConfidences(RFManagerPtr, float[] accNorms, float[] gyroNorms, float[] confidences, int n_classes);

  static {
    Native.register("rr_randomforest");
  }

  protected RFManagerPtr _manager;
  protected int _classCount;

  protected int[] _classLabels;

  public RandomForestAdapter(int sampleCount, int samplingRateHz, String modelFilename) {
    _manager = createRandomForestManager(sampleCount, samplingRateHz, modelFilename);
    _classCount = randomForestGetClassCount();

    _classLabels = new int[_classCount];
    randomForestGetClassLabels(_manager, _classLabels, _classCount);
  }

  public HashMap<Int, Float> predictConfidences(ArrayList<Float> accNorms, ArrayList<Float> gyroNorms) {
    float[] confidences = new float[_classCount];
    randomForestClassificationConfidences(_manager, floatArray(accNorms), floatArray(gyroNorms), confidences, _classCount);
    HashMap<Int, Float> ret = new HashMap<Int, Float>(_classCount);
    for (int i = 0; i < _classCount; ++i) {
      ret.put(_classLabels[i], confidences[i]);
    }

    return ret;
  }

  private float[] floatArray(ArrayList<Float> list) {
    float[] a = new float[list.size()];

    int i = 0;
    for (Float v : list) {
      a[i++] = (v == null ? Float.NaN : v);
    }
    return a;
  }
}
