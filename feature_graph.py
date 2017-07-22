from collections import defaultdict
import numpy as np

from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from pipeline import *

def getFeaturesByLabel(fsets, labels=[], feature_index=0):
    features_by_label = defaultdict(list)
    for fset in fsets:
        if fset.reportedActivityType not in labels:
            continue
        feature_values = [vec[feature_index] for vec in fset.features]
        features_by_label[fset.reportedActivityType] += feature_values

    return features_by_label

def graphFeatures(fsets, labels=[], feature_index=0):
    features_by_label = getFeaturesByLabel(fsets, labels=labels, feature_index=feature_index)

    np_by_label = { label: np.array(values) for label, values in features_by_label.items() }

    least_value = min(np.min(np_by_label[label]) for label in labels)
    greatest_value = max(np.max(np_by_label[label]) for label in labels)
    X_plot = np.linspace(least_value, greatest_value, 100)[:, np.newaxis]

    colors = list('bgrcmykw')

    fig, ax = plt.subplots()
    for index, label in enumerate(labels):
        color = colors[index]
        X = np_by_label[label][:, np.newaxis]
        kde = KernelDensity(kernel='epanechnikov', bandwidth=10)
        kde.fit(X)
        log_density = kde.score_samples(X_plot)
        ax.plot(X_plot, np.exp(log_density), '-{}'.format(color),
                 label='kde {}'.format(label))
        ax.plot(X, -0.005 - 0.012 * index - 0.01 * np.random.random(X.shape[0]),
                 '+{}'.format(color), label='values {}'.format(label), alpha=0.1)

    ax.legend(loc='upper right')
    return plt

def graphFeaturesFromPickles(labels=[], platform=None,
                             seed=None, fraction=1.0, include_crowd_data=False):
    fsets = loadAllFeatureSets(
        platform=platform,
        seed=seed,
        fraction=fraction,
        include_crowd_data=include_crowd_data,
        use_processes=False,
        exclude_labels=[]
    )
    for feature_index in xrange(0, 13):
        plt = graphFeatures(fsets, labels=labels, feature_index=feature_index)
    plt.show()
