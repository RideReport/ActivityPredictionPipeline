from rr_mode_classification import RandomForest
import numpy as np
rf = RandomForest(8, "trained_model.cv")
print "made rf"
features = rf.prepareFeatures(list(np.arange(0., 8., 1.)))
print "features ", features
confidences = rf.predict_proba(list(np.arange(0., 8., 1.)))
print "confidences ", confidences
print "done"
