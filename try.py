from rr_mode_classification import RandomForest
import numpy as np
rf = RandomForest(8, "trained_model.cv")
print "made rf"
features = rf.prepareFeatures(list(np.arange(0., 8., 1.)))
print "features ", features
confidences = rf.predict_proba(list(np.arange(0., 8., 1.)))
print "confidences ", confidences
print "done"

class RFClassifier:

    def __init__(self):
        from rr_mode_classification import RandomForest
        self.forest = RandomForest(32, "trained_model.cv")

    def predict_proba_norms(self, norms):
        accumulator = []
        for window in rolling_window(norms, 32):
            confidences = theForest.predict_proba(list(window))
            accumulator.append(confidences)
        return np.array(accumulator)

    def get_class_labels(self):
        return self.forest.getClassLabels()

