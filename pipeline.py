import numpy as np
import pytz
import datetime
import json
import math
import dateutil.parser
import os
import pickle
import random
from itertools import izip
from contexttimer import Timer
from functools import partial

from multiprocessing import Pool
from tqdm import tqdm
import glob
from operator import itemgetter
from contextlib import contextmanager

READING_X = 0
READING_Y = 1
READING_Z = 2
READING_T = 3

SPEED_S = 0
SPEED_T = 1

class InvalidSampleError(ValueError):
    pass

class IncompatibleFeatureCountError(ValueError):
    pass

class AccelerationVector3D(object):
    epoch = pytz.utc.localize(datetime.datetime.utcfromtimestamp(0))
    def __init__(self, accDicts):
        self._accelerationsSorted = sorted(accDicts, key=itemgetter('date'))
        self._accNormsList = [self.rawNorm(acc) for acc in self._accelerationsSorted]
        self._datesList = [dateutil.parser.parse(acc['date']) for acc in self._accelerationsSorted]
        self._origin = self._datesList[0]
        self._relativeSeconds = [(dt - self._origin).total_seconds() for dt in self._datesList]
        self._epochSeconds = [(dt - self.epoch).total_seconds() for dt in self._datesList]

        # Use the epoch as origin, just as is done in java & swift apps
        self.readings = [
            { 'x': acc['x'], 'y': acc['y'], 'z': acc['z'], 't': second }
            for acc, second in izip(self._accelerationsSorted, self._epochSeconds)
        ]

    @staticmethod
    def rawNorm(row):
        return math.sqrt(row['x']**2 + row['y']**2 + row['z']**2)

    @property
    def norms(self):
        return self._accNormsList

    @property
    def seconds(self):
        return self._relativeSeconds

    @property
    def npReadings(self):
        return np.array([(r['x'], r['y'], r['z'], r['t']) for r in self.readings])

def readingsFromNPReadings(npReadings):
    readings = []
    for npRow in npReadings:
        readings.append({
            'x': npRow[READING_X],
            'y': npRow[READING_Y],
            'z': npRow[READING_Z],
            't': npRow[READING_T],
        })
    return readings

class LocationSpeedVector(object):
    epoch = pytz.utc.localize(datetime.datetime.utcfromtimestamp(0))

    def __init__(self, locationDicts):
        self.npSpeedReadings = np.array([
            (
                loc['speed'],
                (dateutil.parser.parse(loc['date']) - self.epoch).total_seconds()
            )
            for loc in locationDicts])


class RawSample(object):
    def __init__(self, classification_data, pk):
        self.pk = pk
        self.notes = classification_data.get('notes', '')
        self.device = classification_data.get('identifier', '')
        self.reportedActivityType = classification_data.get('reportedActivityType', classification_data.get('data', {}).get('reportedActivityType'))
        if self.reportedActivityType is None:
            raise InvalidSampleError()

        self.npRawReadings = AccelerationVector3D(classification_data['data']['accelerometerAccelerations']).npReadings
        self.npRawSpeedReadings = LocationSpeedVector(classification_data['data']['locations']).npSpeedReadings

class npReadingsMixin(object):
    @property
    def sampleCount(self):
        return len(self.npReadings)

    @property
    def minT(self):
        return np.min(self.npReadings[:,READING_T])

    @property
    def maxT(self):
        return np.max(self.npReadings[:,READING_T])

    @property
    def duration(self):
        return self.maxT - self.minT

    @property
    def maxSpacing(self):
        return np.max(np.diff(self.npReadings[:,READING_T]))

class npSpeedReadingsMixin(object):
    @property
    def averageSpeed(self):
        if self.npSpeedReadings.shape[0] == 0:
            return None
        speeds = self.npSpeedReadings[:,SPEED_S]
        meanSpeed = np.mean(speeds[speeds >= 0])
        if np.isnan(meanSpeed):
            return None
        else:
            return meanSpeed


class ContinuousIntervalSample(npReadingsMixin, npSpeedReadingsMixin):

    MAX_ALLOWED_SPACING = 1./20. * 1.1 # max number of seconds between readings
    MAX_INTERVAL_LENGTH = 60. # max number of seconds for entire interval (for train/test splitting)
    MIN_INTERVAL_LENGTH = 64/20. # minimum length in seconds for a useful interval

    def __init__(self, sample, startIndex, endIndex):
        self.samplePk = sample.pk
        self.sampleNotes = sample.notes
        self.reportedActivityType = sample.reportedActivityType
        self.npReadings = sample.npRawReadings[startIndex:endIndex]

        if sample.npRawSpeedReadings.shape[0] > 0:
            speedTimes = sample.npRawSpeedReadings[:,SPEED_T]
            minT = self.minT
            maxT = self.maxT
            self.npSpeedReadings = sample.npRawSpeedReadings[(speedTimes >= minT) & (speedTimes <= maxT)]
        else:
            self.npSpeedReadings = np.empty((0, 2))

    def __getstate__(self):
        keys = ('samplePk', 'sampleNotes', 'reportedActivityType', 'npReadings', 'npSpeedReadings')
        return { k: v for k, v in self.__dict__.iteritems() if k in keys }

    @classmethod
    def makeMany(cls, sample,
            max_spacing=MAX_ALLOWED_SPACING,
            max_interval=MAX_INTERVAL_LENGTH,
            min_interval=MIN_INTERVAL_LENGTH):

        def makeCiSample(sample, startIndex, index):
            try:
                if index - startIndex > 0:
                    return ContinuousIntervalSample(sample, startIndex, index)
            except:
                import traceback
                print "Skipping sample due to exception during creation"
                traceback.print_exc()
                pass
            return None

        def generateIntervals():
            prevRow = None
            startIndex = 0
            for index, npRow in enumerate(sample.npRawReadings):
                if prevRow is not None:
                    spacing = npRow[READING_T] - prevRow[READING_T]
                    if spacing > max_spacing:
                        yield makeCiSample(sample, startIndex, index)
                        startIndex = index
                    elif (npRow[READING_T] - sample.npRawReadings[startIndex][READING_T]) > max_interval:
                        yield makeCiSample(sample, startIndex, index)
                        startIndex = index
                prevRow = npRow

            yield makeCiSample(sample, startIndex, index)

        def isAcceptable(ciSample):
            return ciSample is not None and ciSample.duration >= min_interval

        return list(ciSample for ciSample in generateIntervals() if isAcceptable(ciSample))

class LabeledFeatureSet(object):

    @classmethod
    def fromTSDEvents(cls, reportedActivityType, events, forest):
        fset = cls()
        fset.reportedActivityType = reportedActivityType
        fset.features = []
        for event in events:
            try:
                fset.features.append(event.getFeatures(forest))
            except RuntimeError:
                pass
        return fset

    @classmethod
    def fromSample(cls, ciSample, forest, spacing):
        fset = cls()
        fset.samplePk = ciSample.samplePk
        fset.reportedActivityType = ciSample.reportedActivityType
        fset.features = list(fset._generateFeatures(ciSample, forest, spacing))
        return fset

    @classmethod
    def _generateFeatures(cls, ciSample, forest, spacing):
        """Generate features based on a continuous sample.

        It is OK to do a rolling window here because the test/train split
        is done when these ciSample objects are created
        """
        try:
            offsetIndex = 0
            while True:
                offset = offsetIndex * spacing
                minT = ciSample.minT + offset
                maxT = minT + forest.desired_signal_duration*1.5
                readingTimes = ciSample.npReadings[:,READING_T]
                mask = (readingTimes >= minT) & (readingTimes <= maxT)

                # convert to dict because that's what C++ wrapper expects
                readings = readingsFromNPReadings(ciSample.npReadings[mask])

                yield forest.prepareFeaturesFromSignal(readings)
                offsetIndex += 1
        except RuntimeError as e:
            if 'probably not enough data' in str(e):
                pass
            else:
                raise

class ModelBuilder(object):

    def __init__(self, feature_sets, sample_count_multiple=.0005, active_var_count=0, max_tree_count=10, epsilon=0.0001):
        self.sample_count_multiple = sample_count_multiple
        self.active_var_count = active_var_count
        self.max_tree_count = max_tree_count
        self.epsilon = epsilon

        self.features = None
        self.labels = None
        self._appendFeaturesAndLabels(*self._convertFeatureSetsToFeaturesAndLabels(feature_sets))

    @staticmethod
    def _convertFeatureSetsToFeaturesAndLabels(fset_list):
        def gen_feature_counts(fset_list):
            for fset in fset_list:
                for feature_row in fset.features:
                    yield len(feature_row)
        feature_counts = set(gen_feature_counts(fset_list))
        if len(feature_counts) > 1:
            print feature_counts
            raise IncompatibleFeatureCountError()
        feature_count = feature_counts.pop()

        all_features = np.array(sum((fset.features for fset in fset_list), []), dtype=np.float32)
        all_labels = np.empty(len(all_features), dtype=np.int32)

        startIndex = 0
        for fset in fset_list:
            all_labels[startIndex:startIndex+len(fset.features)] = fset.reportedActivityType
            startIndex += len(fset.features)

        return (all_features, all_labels)

    def _appendFeaturesAndLabels(self, features, labels):
        if self.features is None:
            self.features = features
            self.labels = labels
        else:
            self.features = np.append(self.features, features)
            self.labels = np.append(self.labels, labels)

    def build(self, output_filename="./data/trained_model.cv"):
        import cv2
        classes = np.unique(self.labels)
        varIdx = np.ones(self.features.shape[1], dtype=np.uint8)
        sampleIdx = np.ones(self.features.shape[0], dtype=np.uint8)
        varTypes = np.array([cv2.ml.VAR_NUMERICAL] * self.features.shape[1] + [cv2.ml.VAR_CATEGORICAL], dtype=np.uint8)

        traindata = cv2.ml.TrainData_create(
            samples=self.features,
            layout=cv2.ml.ROW_SAMPLE,
            responses=self.labels,
            varIdx=varIdx,
            sampleIdx=sampleIdx,
            varType=varTypes)

        model = cv2.ml.RTrees_create()
        model.setMaxDepth(30)
        model.setCVFolds(1) # higher numbers crash
        # model.setTruncatePrunedTree(False)
        # model.setUse1SERule(False)
        model.setActiveVarCount(self.active_var_count)
        #model.setMinSampleCount(int(self.features.shape[0] * 0.005))
        #sample_count = 10
        sample_count = int(self.features.shape[0] * self.sample_count_multiple)
        print "Sample count: {}".format(sample_count)
        model.setMinSampleCount(sample_count)
        # model.setTruncatePrunedTree(True) # reduce output size?

        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, self.max_tree_count, self.epsilon))
        model.setCalculateVarImportance(True)
        n_samples = self.features.shape[0]
        priors = np.array([n_samples/(len(classes) * float(count)) for cls, count in izip(*np.unique(self.labels, return_counts=True))])
        print 'Setting priors: '
        for label, prior in izip(classes, priors):
            print "{: 2}: {: 4.1f}".format(label, prior)
        model.setPriors(priors / np.linalg.norm(priors))
        print "Beginning training with {} rows of features".format(self.features.shape[0])
        with Timer() as t:
            model.train(traindata)
        print "Trained in {:.1f}s".format(t.elapsed)

        model.save(output_filename)
        print "CV model: {} ({:.4f} MB)".format(output_filename, os.path.getsize(output_filename) / (2.**20))

        return BuiltForest(model=model)

class BuiltForest(object):
    def __init__(self, model=None, params={}):
        self.model = model
        self.params = params

    def printConfusionMatrix(self, test_feature_sets):
        features, labels = ModelBuilder._convertFeatureSetsToFeaturesAndLabels(test_feature_sets)

        for label in np.unique(labels):
            junk, predicted_labels = self.model.predict(features[labels == label])
            total = float(predicted_labels.shape[0])
            accuracy = predicted_labels[predicted_labels == label].shape[0] / total
            wrongs = ['{}={:2}'.format(int(cls), int(100* predicted_labels[predicted_labels==cls].shape[0]/total)) for cls in np.unique(labels)]
            print "CV Accuracy {}: {:3} {} ({: 6} total)".format(label, int(100*accuracy), ' '.join(wrongs), int(total))


class TSDEvent(npReadingsMixin, npSpeedReadingsMixin):
    def __init__(self, event_dict):
        if 'activityTypePredictions' not in event_dict or len(event_dict['activityTypePredictions']) == 0:
            self.originalInferredActivityType = None
        else:
            old_predictions = { p['activityType']: p['confidence'] for p in event_dict['activityTypePredictions'] }
            self.originalInferredActivityType = max(old_predictions.keys(), key=lambda k: old_predictions[k])

        self.npReadings = AccelerationVector3D(event_dict['accelerometerAccelerations']).npReadings
        self.npSpeedReadings = LocationSpeedVector(event_dict['locations']).npSpeedReadings

    def getFreshInferredType(self, forest):
        try:
            confidences = forest.classifySignal(readingsFromNPReadings(event.npReadings))
        except RuntimeError:
            return None
        predictions = { k: v for k, v in izip(forest.classLabels(), confidences) }
        return max(predictions.keys(), key=lambda k: predictions[k])

    def getFeatures(self, forest):
        return forest.prepareFeaturesFromSignal(readingsFromNPReadings(self.npReadings))

class PreparedTSD(object):
    def __init__(self, tsd_dict):
        self.notes = tsd_dict.get('notes', '')
        self.reportedActivityType = tsd_dict['reported_type']
        self.events = [TSDEvent(event) for event in tsd_dict['data']]

@contextmanager
def print_exceptions():
    try:
        yield
    except:
        import traceback
        traceback.print_exc()
        raise

def updateContinuousIntervalsPickleFromJSONFile(filename):
    with print_exceptions():
        try:
            with open(filename) as f:
                data = json.load(f)
        except:
            print "Could not load data from file: {}".format(filename)
            return 0

        sample = RawSample(data['classification_data'], data['pk'])
        ciSamples = ContinuousIntervalSample.makeMany(sample)
        with open('{}.ciSamples.pickle'.format(filename), 'wb') as f:
            pickle.dump(ciSamples, f, pickle.HIGHEST_PROTOCOL)
        return len(ciSamples)

def derivativeIsOld(filename, derivativePattern):
    try:
        return os.path.getmtime(derivativePattern.format(filename)) < os.path.getmtime(filename)
    except:
        return True # one or both doesn't exist

def updateSamplePickles(force_update=False):
    pool = Pool()
    all_filenames = list(glob.glob("data/classification_data.*.jsonl"))
    filenames = [fname for fname in all_filenames if force_update or derivativeIsOld(fname, '{}.ciSamples.pickle')]
    overall_sample_count = 0
    with Timer() as t:
        for sample_count in tqdm(pool.imap_unordered(updateContinuousIntervalsPickleFromJSONFile, filenames), total=len(filenames)):
            overall_sample_count += sample_count
    print "Completed {} samples from {}/{} files in {:.1f}s".format(overall_sample_count, len(filenames), len(all_filenames), t.elapsed)

def updateFeatureSetsPickleFromCiSamplesPickle(filename):
    with print_exceptions():
        from rr_mode_classification_opencv import RandomForest
        sampleCount = 64
        samplingRateHz = 20
        forest = RandomForest(sampleCount, samplingRateHz, None)
        with open(filename) as f:
            ciSamples = pickle.load(f)

        fset_list = []
        for ciSample in ciSamples:
            fset = LabeledFeatureSet.fromSample(ciSample, forest, 1./samplingRateHz)
            fset_list.append(fset)

        with open('{}.fsets.pickle'.format(filename), 'wb') as f:
            pickle.dump(fset_list, f, pickle.HIGHEST_PROTOCOL)
        return sum(len(fset.features) for fset in fset_list)

def updateFeatureSets(force_update=False):
    from multiprocessing import Pool
    import glob
    import pickle
    from tqdm import tqdm
    pool = Pool()

    all_filenames = glob.glob('./data/*.ciSamples.pickle')
    filenames = [fname for fname in all_filenames if force_update or derivativeIsOld(fname, '{}.fsets.pickle')]
    overall_feature_count = 0
    with Timer() as t:
        for feature_count in tqdm(pool.imap_unordered(updateFeatureSetsPickleFromCiSamplesPickle, filenames), total=len(filenames)):
            overall_feature_count += feature_count

    print "Generated {} rows of features from {}/{} files in {:.1f}s".format(feature_count, len(filenames), len(all_filenames), t.elapsed)

def getFeatureSetsFromAllTrainableTSDs():
    filenames = glob.glob('./data/trusted_tsd.*.jsonl')
    pool = Pool()
    return list(tqdm(pool.imap_unordered(getFeatureSetFromTrainableTSDFile, filenames), total=len(filenames)))

def getReportedActivityTypeWithOverrides(tsd):
    if tsd.reportedActivityType == 4 and 'run' in tsd.notes:
        print "Overriding {} -> {}: {}".format(tsd.reportedActivityType, 1, tsd.notes)
        return 1
    return tsd.reportedActivityType

def getFeatureSetFromTrainableTSDFile(filename):
    with print_exceptions():
        from rr_mode_classification_opencv import RandomForest
        forest = RandomForest(64, 20, None)
        tsd = loadTSD(filename, force_update=True)
        activityType = tsd.reportedActivityType

        return LabeledFeatureSet.fromTSDEvents(tsd.reportedActivityType, tsd.events, forest)

def buildModelFromFeatureSetPickles(output_filename, split, exclude_labels, include_crowd_data=False):
    all_sets = []

    if include_crowd_data:
        print "Loading whitelisted TSDs"
        all_sets += getFeatureSetsFromAllTrainableTSDs()

    filenames = glob.glob('./data/*.fsets.pickle')
    with Timer() as t:
        for filename in filenames:
            with open(filename) as f:
                sets = pickle.load(f)
                if len(sets) > 0 and sets[0].reportedActivityType not in exclude_labels:
                    all_sets += sets
    print "Loaded {} feature sets from {} files in {:.1f}s".format(len(all_sets), len(filenames), t.elapsed)

    if split:
        bytype = {}
        for fset in all_sets:
            bytype.setdefault(fset.reportedActivityType, [])
            bytype[fset.reportedActivityType].append(fset)

        train_sets = []
        test_sets = []
        for ksets in bytype.values():
            random.shuffle(ksets)

            train_sets += ksets[:len(ksets)/2]
            test_sets += ksets[len(ksets)/2:]

        builder = ModelBuilder(train_sets)
    else:
        builder = ModelBuilder(all_sets)

    builtforest = builder.build(output_filename=output_filename)

    if split:
        builtforest.printConfusionMatrix(test_sets)
    else:
        print "Cannot print confusion matrix without splitting"

def loadTSD(filename, force_update=False):

    pickleFilename = '{}.pickle'
    try:
        pickleTime = os.path.getmtime(pickleFilename)
    except:
        pickleTime = 0

    fileTime = os.path.getmtime(filename)
    if fileTime > pickleTime or force_update:
        with open(filename) as tsdF:
            tsd = PreparedTSD(json.load(tsdF))
        with open(pickleFilename, 'wb') as pickleF:
            pickle.dump(tsd, pickleF, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(pickleFilename) as pickleF:
                tsd = pickle.load(pickleF)
        except:
            return loadTSD(filename, force_update=True)

    return tsd

def getFeaturesAndLabelsFromTSD(tsd, forest):
    reportedActivityType = tsd.reportedActivityType
    features = []
    for event in tsd.events:
        features.append(tsd.getFeatures(forest))
        labels.append(tsd.reportedActivityType)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

def predictTSDFileWithFilters(forestPath, filename):
    from rr_mode_classification_opencv import RandomForest
    with print_exceptions():
        forest = RandomForest(64, 20, forestPath)
        classLabels = forest.classLabels()
        tsd = loadTSD(filename)

        confusion_tuples = []
        reportedActivityType = tsd.reportedActivityType
        for event in tsd.events:
            # skip if original predictions aren't known
            if event.originalInferredActivityType is None:
                continue

            # skip if speed is unknown
            if event.averageSpeed is None or event.averageSpeed < 0:
                continue

            # skip if wrong speed for type
            speed = event.averageSpeed
            running_slow = (reportedActivityType == 1 and speed < 1.5)
            bike_or_motor = (reportedActivityType == 2 or reportedActivityType == 3 or reportedActivityType == 5 or reportedActivityType == 6)
            bike_or_motor_slow = bike_or_motor and speed < 2
            walking_slow = (reportedActivityType == 4 and speed < 0.3)
            walking_fast = (reportedActivityType == 4 and speed > 2)
            if speed < 0 or running_slow or bike_or_motor_slow or walking_fast or walking_slow:
                continue

            # create new predictions
            confusion_tuples.append((reportedActivityType, event.originalInferredActivityType, event.getFreshInferredType(forest)))
        return confusion_tuples

def loadModelAndTestAgainstTSDs(forestPath, fraction=1.0):
    import glob
    import random
    import os
    from tqdm import tqdm
    from multiprocessing import Pool
    from rr_mode_classification_opencv import RandomForest

    forest = RandomForest(64, 20, forestPath)
    if not forest.canPredict():
        print "model at '{}' Cannot predict!".format(forestPath)
        return

    classLabels = forest.classLabels()

    def printConfusion(confusion):
        labels = [1, 2, 3, 4, 5, 6, 7]
        print '   {}  TOTAL'.format(' '.join("{: 4}".format(column_title) for column_title in labels))
        for reported_label in labels:
            print "{: 2}".format(reported_label),
            total = sum(v for k, v in confusion.items() if k[0] == reported_label)
            total = max(total, 1)
            for predicted_label in labels:
                k = (reported_label, predicted_label)
                print "{: 4.0f}".format(confusion.get(k, 0) / float(total) * 100),
            print "{: 6}".format(total)

    fresh_confusion = {}
    old_confusion = {}
    with Timer() as t:
        pool = Pool()
        tsd_files = glob.glob('./data/tsd*')
        if fraction < 1.0:
            tsd_files = random.sample(tsd_files, int(len(tsd_files)*fraction))

        # big files first, for better parallelization
        sorted_files = sorted(tsd_files, key=lambda filename: os.path.getsize(filename), reverse=True)

        prediction_count = 0
        for predictions in tqdm(pool.imap_unordered(partial(predictTSDFileWithFilters, forestPath), sorted_files), total=len(sorted_files)):
            for reported_type, old_type, fresh_type in predictions:
                prediction_count += 1

                confusion_key = (reported_type, fresh_type)
                fresh_confusion.setdefault(confusion_key, 0)
                fresh_confusion[confusion_key] += 1

                confusion_key = (reported_type, old_type)
                old_confusion.setdefault(confusion_key, 0)
                old_confusion[confusion_key] += 1

    print "Got {} predictions from {} TSDs in {:.1f}s".format(prediction_count, len(tsd_files), t.elapsed)

    print "Original prediction confusion:"
    printConfusion(old_confusion)

    print "New model confusion:"
    printConfusion(fresh_confusion)

def dispatchCommand(command, options):
    if command == 'updateSamples':
        updateSamplePickles(force_update=options.force_update)
    elif command == 'updateFeatures':
        updateFeatureSets(force_update=options.force_update)
    elif command == 'train':
        try:
            exclude_labels = [int(s) for s in options.exclude_labels.split(',')]
        except:
            exclude_labels = []
        buildModelFromFeatureSetPickles(
            output_filename=options.output_filename,
            split=options.split,
            exclude_labels=exclude_labels,
            include_crowd_data=options.include_crowd_data)
    elif command == 'test':
        loadModelAndTestAgainstTSDs(options.model, fraction=options.tsd_sample_fraction)
    elif command == 'all':
        dispatchCommand('updateSamples', options)
        dispatchCommand('updateFeatures', options)
        dispatchCommand('train', options)
        dispatchCommand('test', options)
    else:
        raise ValueError("Unknown command: {}".format(command))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="run a pipeline command")
    parser.add_argument('command', metavar='CMD', type=str)
    parser.add_argument('-f --force', dest='force_update', action='store_true', default=False)
    parser.add_argument('-m --model', dest='model', type=str, default='./model.cv')
    parser.add_argument('-o', '--output', dest='output_filename', type=str, default='./model.cv')
    parser.add_argument('--exclude-labels', dest='exclude_labels', type=str, default='9')
    parser.add_argument('--no-split', dest='split', default=True, action='store_false')
    parser.add_argument('--sample-fraction', dest='tsd_sample_fraction', default=1.0, type=float)
    parser.add_argument('--include-crowd-data', dest='include_crowd_data', action='store_false', default=True)
    args = parser.parse_args()
    dispatchCommand(args.command, args)
