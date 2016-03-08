from rr_mode_classification import RandomForest
import numpy as np
rf = RandomForest(8, "trained_model.cv")
print "made rf"
features = rf.prepareFeatures(list(np.arange(0., 8., 1.)))
rf.classify(list(np.arange(0., 8., 1.)))
print features
print "done"
