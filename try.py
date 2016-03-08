import main
import numpy as np
rf = main.RandomForest(8, "trained_model.cv")
print "made rf"
features = rf.prepareFeatures(list(np.arange(0., 8., 1.)))
rf.classify(list(np.arange(0., 8., 1.)))
print features
print "done"
