import numpy as np
import pytz
import datetime
import json
import yaml
import math
import dateutil.parser
import git
import os
import pickle
import random
import hashlib
import inspect
from copy import copy
from itertools import izip
from contexttimer import Timer
from functools import partial


from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
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

SAMPLING_RATE_HZ = 21
SAMPLE_COUNT = 64

class InvalidSampleError(ValueError):
    pass

class IncompatibleFeatureCountError(ValueError):
    pass

class IncompatibleActivityTypesError(ValueError):
    pass

class IncompatibleConfigurationError(ValueError):
    pass

class NonPredictingForestError(ValueError):
    pass

@contextmanager
def terminatingPool(use_processes=True):
    if use_processes:
        pool = Pool()
    else:
        pool = ThreadPool()

    try:
        yield pool
    finally:
        pool.terminate()

def sha256_sorted_json(thing):
    return hashlib.sha256(json.dumps(thing, sort_keys=True)).hexdigest()

def filename_sha256_hexdigest(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def ordered_predictions_str(labels, confidences=None):
    if confidences is not None:
        predictions_list = sorted(izip(labels, confidences), key=itemgetter(1), reverse=True)
    else:
        assert isinstance(labels, dict)
        predictions_list = sorted(labels.items(), key=itemgetter(1), reverse=True)
    return ' '.join('{}={:.1f}'.format(k, v) for k, v in predictions_list)

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

class TSDConfusionAccumulator(object):
    def __init__(self):
        self.old = {}
        self.fresh = {}
        self.count = 0

    def add(self, prediction):
        reported_type, old_type, fresh_type = prediction

        self.count += 1

        k = (reported_type, old_type)
        self.old.setdefault(k, 0)
        self.old[k] += 1

        k = (reported_type, fresh_type)
        self.fresh.setdefault(k, 0)
        self.fresh[k] += 1


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
        return self.npReadings.shape[0]

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

    @property
    def minSpacing(self):
        return np.min(np.diff(self.npReadings[:,READING_T]))

class npSpeedReadingsMixin(object):
    @property
    def averageSpeed(self):
        if self.npSpeedReadings.shape[0] == 0:
            return None
        speeds = self.npSpeedReadings[:,SPEED_S]
        good_speeds = speeds[speeds >= 0]
        if len(good_speeds):
            return np.mean(good_speeds)
        else:
            return None


class ContinuousIntervalSample(npReadingsMixin, npSpeedReadingsMixin):

    MAX_ALLOWED_SPACING = 1./float(SAMPLING_RATE_HZ) * 1.1 # max number of seconds between readings
    MAX_INTERVAL_LENGTH = 60. # max number of seconds for entire interval (for train/test splitting)
    MIN_INTERVAL_LENGTH = SAMPLE_COUNT/float(SAMPLING_RATE_HZ) # minimum length in seconds for a useful interval

    def __init__(self, sample, startIndex, endIndex):
        self.samplePk = sample.pk
        self.dataId = { 'type': 'ciSample', 'pk': sample.pk, 'startIndex': startIndex, 'endIndex': endIndex }
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

    def generateSubSampleMasks(self, spacing, desired_duration):
        offsetIndex = 0
        previousStartIndex = -1
        previousAfterEndIndex = -1
        while True:
            offset = offsetIndex * spacing

            minT = self.minT + offset
            maxT = minT + desired_duration
            readingTimes = self.npReadings[:,READING_T]
            mask = (readingTimes >= minT) & (readingTimes <= maxT)
            startIndex = np.argmax(mask)

            if startIndex != previousStartIndex:
                yield mask

            # Get index of first True value, starting from the end of the mask.
            # If this is zero, then we're done.
            indicesToRight = np.argmax(mask[::-1])
            if indicesToRight == 0:
                return

            offsetIndex += 1
            previousStartIndex = startIndex

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
    def fromTSD(cls, tsd, forest, event_indices=None):
        fset = cls()
        fset.reportedActivityType = tsd.reportedActivityType
        fset.features = []
        indices = []
        for index, event in enumerate(tsd.events):
            try:
                if event_indices is None or index in event_indices:
                    fset.features.append(event.getFeatures(forest))
                    indices.append(index)
            except RuntimeError:
                pass
        fset.dataId = copy(tsd.dataId)
        fset.dataId['indices'] = indices
        return fset

    @classmethod
    def fromSample(cls, ciSample, forest, forestConfigStr, rolling_window_spacing):
        fset = cls()
        fset.samplePk = ciSample.samplePk
        fset.dataId = copy(ciSample.dataId)
        fset.reportedActivityType = ciSample.reportedActivityType
        fset.features = list(fset._generateFeatures(ciSample, forest, rolling_window_spacing))
        fset.forestConfigStr = forestConfigStr
        fset.forestConfigHash = sha256_sorted_json(json.loads(forestConfigStr))
        return fset

    @classmethod
    def _generateFeatures(cls, ciSample, forest, spacing):
        """Generate features based on a continuous sample.

        It is OK to do a rolling window here because the test/train split
        is done on the LabeledFeatureSet level and every overlapping sample
        generated here will be recorded to the same LabeledFeatureSet.
        """

        # construct sub-samping window of desired duration, plus 10%
        # for safety
        for mask in ciSample.generateSubSampleMasks(spacing=spacing, desired_duration=forest.desired_signal_duration*1.1):
            readings = readingsFromNPReadings(ciSample.npReadings[mask])
            try:
                yield forest.prepareFeaturesFromSignal(readings)
            except RuntimeError as e:
                if 'probably not enough data' in str(e):
                    break
                else:
                    raise

class ModelBuilder(object):

    def __init__(self, feature_sets, sample_count_multiple=.0005, active_var_count=0, max_tree_count=10, epsilon=0.0001):
        self.sample_count_multiple = sample_count_multiple
        self.active_var_count = active_var_count
        self.max_tree_count = max_tree_count
        self.epsilon = epsilon

        self.dataId = sorted([fset.dataId for fset in feature_sets], key=sha256_sorted_json)

        configHashes = set(fset.forestConfigHash for fset in feature_sets)
        if len(configHashes) != 1:
            raise IncompatibleConfigurationError(repr(configHashes))
        self.forestConfigStr = next(fset.forestConfigStr for fset in feature_sets)
        self.forestConfigHash = configHashes.pop()

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

        return BuiltForest(builder=self, model=model, filename=output_filename)

class BuiltForest(object):
    def __init__(self, builder=None, model=None, filename=None):
        self.builder = builder
        self.model = model
        self.filename = filename

        self.cv_meta = self._loadMetaFromCvFile(filename)

    @classmethod
    def _loadMetaFromCvFile(cls, filename):
        text = u''
        with open(filename) as f:
            f.readline() # skip first line
            for line in f:
                if line.strip() == 'trees:':
                    # PyYaml can't parse the forest data generated by OpenCV,
                    # and we don't need the actual splits here, so we stop reading
                    break

                # This is a type-hint that we don't care about.
                line = line.replace('!!opencv-matrix', '')
                text += line

        data = yaml.load(text)
        return data

    def getConfusionMatrixStr(self, test_feature_sets):
        features, labels = ModelBuilder._convertFeatureSetsToFeaturesAndLabels(test_feature_sets)

        lines = []
        for label in np.unique(labels):
            junk, predicted_labels = self.model.predict(features[labels == label])
            total = float(predicted_labels.shape[0])
            accuracy = predicted_labels[predicted_labels == label].shape[0] / total
            wrongs = ['{}={:2}'.format(int(cls), int(100* predicted_labels[predicted_labels==cls].shape[0]/total)) for cls in np.unique(labels)]
            lines.append("CV Accuracy {}: {:3} {} ({: 6} total)".format(label, int(100*accuracy), ' '.join(wrongs), int(total)))
        self.matrix = '\n'.join(lines)
        self.matrix_features_dataId = sorted([fset.dataId for fset in test_feature_sets], key=sha256_sorted_json)
        return self.matrix

    def tuple_sha256_hexdigest(self, t):
        sha256 = hashlib.sha256()
        for value in t:
            sha256.update(json.dumps(value))
        return sha256.hexdigest()

    def metadata(self, **kwargs):
        thisRepo = git.Repo(os.path.abspath(os.path.dirname(__file__)), search_parent_directories=True)
        predictorRepo = git.Repo(os.path.abspath(os.path.join(os.path.dirname(__file__), 'ActivityPredictor')), search_parent_directories=True)
        data = dict(
            model_metadata_version=1,
            features=dict(
                config_sha256=self.builder.forestConfigHash,
                count=len(self.builder.labels),
            ),
            builder=dict(
                sample_count_multiple=self.builder.sample_count_multiple,
                active_var_count=self.builder.active_var_count,
                max_tree_count=self.builder.max_tree_count,
                epsilon=self.builder.epsilon,
            ),
            extra=kwargs,
            cv_sha256=filename_sha256_hexdigest(self.filename),
            cv=self.cv_meta,
            data=dict(
                sha256=sha256_sorted_json(self.builder.dataId),
                id=self.builder.dataId,
            ),
            code=dict(
                pipeline_commit_sha=thisRepo.head.object.hexsha,
                pipeline_is_dirty=thisRepo.is_dirty(),
                predictor_commit_sha=predictorRepo.head.object.hexsha,
                predictor_is_dirty=predictorRepo.is_dirty(),
            ),
            confusion=dict(
                matrix_is_oob=kwargs['split'],
                matrix_str=self.matrix,
                matrix_features_id=self.matrix_features_dataId,
            )
        )

        # merge forest configuration used for features into metadata
        forestConfig = json.loads(self.builder.forestConfigStr)
        for key in forestConfig:
            if isinstance(forestConfig[key], dict):
                data.setdefault(key, {})
                data[key].update(forestConfig[key])
            else:
                data[key] = forestConfig[key]
        return data

    def metadata_brief(self, **kwargs):
        data = self.metadata(**kwargs)
        del data['data']['id']
        del data['confusion']
        del data['cv']
        return data

    def save_to_dir(self, dirname, **kwargs):
        recipe = self.metadata(**kwargs)
        config = self.metadata_brief(**kwargs)
        with open(os.path.join(dirname, 'config.json'), 'wb') as f:
            f.write(json.dumps(config, indent=2, separators=(',', ': '), sort_keys=True))

        with open(os.path.join(dirname, 'recipe.json'), 'wb') as f:
            f.write(json.dumps(recipe, indent=2, separators=(',', ': '), sort_keys=True))

        cv_filename = '{}.cv'.format(recipe['cv_sha256'])
        os.rename(self.filename, os.path.join(dirname, cv_filename))

        # Delete other entries
        for entryname in os.listdir(dirname):
            if entryname not in ('config.json', 'recipe.json', cv_filename):
                os.remove(os.path.join(dirname, entryname))


class TSDEvent(npReadingsMixin, npSpeedReadingsMixin):
    def __init__(self, event_dict):
        if 'activityTypePredictions' not in event_dict or len(event_dict['activityTypePredictions']) == 0:
            self.originalInferredActivityType = None
        else:
            old_predictions = { p['activityType']: p['confidence'] for p in event_dict['activityTypePredictions'] }
            self.originalPredictions = old_predictions
            self.originalInferredActivityType = max(old_predictions.keys(), key=lambda k: old_predictions[k])

        self.npReadings = AccelerationVector3D(event_dict['accelerometerAccelerations']).npReadings
        self.npSpeedReadings = LocationSpeedVector(event_dict['locations']).npSpeedReadings

    def getFreshInferredType(self, forest):
        try:
            confidences = forest.classifySignal(readingsFromNPReadings(self.npReadings))
        except RuntimeError:
            return None
        predictions = { k: v for k, v in izip(forest.classLabels(), confidences) }
        return max(predictions.keys(), key=lambda k: predictions[k])

    def getFeatures(self, forest):
        return forest.prepareFeaturesFromSignal(readingsFromNPReadings(self.npReadings))

    def describe(self, forest):
        delta = self.duration - forest.desired_signal_duration
        notes = []
        notes.append('Original inference: {}'.format(self.originalInferredActivityType))

        try:
            notes.append('Original predictions: {}'.format(ordered_predictions_str(self.originalPredictions)))
        except AttributeError:
            notes.append('No known original predictions.')
        if self.averageSpeed is None:
            notes.append('Unknown speed')
        else:
            notes.append('Speed: {:.1f} m/s == {:.1f} mi/h'.format(self.averageSpeed, self.averageSpeed * 2.23694))
        notes.append('Sample has {} readings, avg spacing {:.1f}ms'.format(self.sampleCount, (self.duration / self.sampleCount) * 1000.))
        notes.append('Min spacing: {:.1f}ms'.format(self.minSpacing * 1000.))

        if delta < 0:
            notes.append('Sample not long enough: need {:.1f}ms more data'.format(-delta * 1000.))
        else:
            notes.append('Sample good length: has {:.1f}ms more than needed'.format(delta * 1000.))

        if self.maxSpacing > forest.desired_spacing * 1.1:
            notes.append('Sample has large gap; max spacing: {:.1f}ms ({:.0%} of desired)'.format(self.maxSpacing * 1000., self.maxSpacing / forest.desired_spacing))
        elif self.maxSpacing > forest.desired_spacing:
            notes.append('Sample has OK gaps;   max spacing: {:.1f}ms ({:.0%} of desired)'.format(self.maxSpacing * 1000., self.maxSpacing / forest.desired_spacing))
        else:
            notes.append('Sample has good gaps; max spacing: {:.1f}ms ({:.0%} of desired)'.format(self.maxSpacing * 1000., self.maxSpacing / forest.desired_spacing))

        try:
            features = forest.prepareFeaturesFromSignal(readingsFromNPReadings(self.npReadings))
            notes.append('Got features: {}'.format(' '.join('{:.1f}'.format(val) for val in features)))
        except RuntimeError as e:
            notes.append('Could not generate features: "{}"'.format(e))

        try:
            confidences = forest.classifySignal(readingsFromNPReadings(self.npReadings))
            notes.append('Got predictions: {}'.format(ordered_predictions_str(forest.classLabels(), confidences)))
        except RuntimeError:
            notes.append('Could not classify: "{}"'.format(e))
        return notes



class PreparedTSD(object):
    def __init__(self, tsd_dict, dataHash=None):
        self.dataId = { 'type': 'tsd', 'trip_pk': tsd_dict['trip_pk'], 'modified': tsd_dict['modified'] }
        self.trip_pk = tsd_dict['trip_pk']
        self.notes = tsd_dict.get('notes', '')
        if self.notes is None:
            self.notes = ''
        self.reportedActivityType = tsd_dict['reported_type']
        self.predictedActivityType = tsd_dict.get('predicted_type', None)
        self.client = tsd_dict.get('client', {})

        try:
            if tsd_dict['client']['name'] == 'RR':
                self.platform = 'ios'
            elif tsd_dict['client']['name'] == 'RRA':
                self.platform = 'android'
        except (KeyError, TypeError):
            self.platform = 'unknown'

        self.skipped_event_count = 0
        self.events = []
        for event in tsd_dict['data']:
            try:
                self.events.append(TSDEvent(event))
            except IndexError:
                self.skipped_event_count += 1

@contextmanager
def print_exceptions():
    try:
        yield
    except:
        import traceback
        traceback.print_exc()
        raise

def split_fsets(fsets, seed=None):
    if seed:
        random.seed(seed)

    bytype = {}
    for fset in fsets:
        bytype.setdefault(fset.reportedActivityType, [])
        bytype[fset.reportedActivityType].append(fset)

    train_sets = []
    test_sets = []
    for ksets in bytype.values():
        random.shuffle(ksets)

        train_sets += ksets[:len(ksets)/2]
        test_sets += ksets[len(ksets)/2:]
    return train_sets, test_sets

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
    print "Updating data/*.ciSamples.pickle..."
    pool = Pool()
    all_filenames = list(glob.glob("data/classification_data.*.jsonl"))
    filenames = [fname for fname in all_filenames if force_update or derivativeIsOld(fname, '{}.ciSamples.pickle')]
    overall_sample_count = 0
    with Timer() as t:
        for sample_count in tqdm(pool.imap_unordered(updateContinuousIntervalsPickleFromJSONFile, filenames), total=len(filenames)):
            overall_sample_count += sample_count
    pool.close()
    print "Completed {} samples from {}/{} files in {:.1f}s".format(overall_sample_count, len(filenames), len(all_filenames), t.elapsed)

def updateFeatureSetsPickleFromCiSamplesPickle(platform, forestConfigStr, filename):
    with print_exceptions():
        if platform == 'android':
            from rr_mode_classification_opencv import RandomForest
        elif platform == 'ios':
            from rr_mode_classification_apple import RandomForest

        forest = RandomForest(forestConfigStr)
        with open(filename) as f:
            ciSamples = pickle.load(f)

        fset_list = []
        for ciSample in ciSamples:
            fset = LabeledFeatureSet.fromSample(ciSample, forest, forestConfigStr, forest.desired_spacing)
            fset_list.append(fset)

        with open('{}.fsets.{}.pickle'.format(filename, platform), 'wb') as f:
            pickle.dump(fset_list, f, pickle.HIGHEST_PROTOCOL)
        return sum(len(fset.features) for fset in fset_list)

def updateFeatureSets(force_update=False, platform='android', config=''):
    from multiprocessing import Pool
    import glob
    import pickle
    from tqdm import tqdm
    pool = Pool()

    print "Updating data/*.fsets.{}.pickle ...".format(platform)
    print "Using forest config: {}".format(config)

    all_filenames = glob.glob('./data/*.ciSamples.pickle')
    filenames = [fname for fname in all_filenames if force_update or derivativeIsOld(fname, '{}.fsets.{}.pickle'.format('{}', platform))]
    overall_feature_count = 0
    with Timer() as t:
        for feature_count in tqdm(pool.imap_unordered(partial(updateFeatureSetsPickleFromCiSamplesPickle, platform, config), filenames), total=len(filenames)):
            overall_feature_count += feature_count

    pool.close()
    print "Generated {} rows of features from {}/{} files in {:.1f}s".format(overall_feature_count, len(filenames), len(all_filenames), t.elapsed)

def getFeatureSetsFromAllTrainableTSDs(platform, config='', use_processes=True):
    filenames = glob.glob('./data/trusted_tsd.*.jsonl')
    with terminatingPool(use_processes) as pool:
        return list(tqdm(pool.imap_unordered(partial(getFeatureSetFromTrainableTSDFile, platform, config), filenames), total=len(filenames)))

def getReportedActivityTypeWithOverrides(tsd):
    if tsd.reportedActivityType == 4 and 'run' in tsd.notes:
        print "Overriding {} -> {}: {}".format(tsd.reportedActivityType, 1, tsd.notes)
        return 1
    return tsd.reportedActivityType

def getFeatureSetFromTrainableTSDFile(platform, forestConfigStr, filename):
    with print_exceptions():
        if platform == 'android':
            from rr_mode_classification_opencv import RandomForest
        elif platform == 'ios':
            from rr_mode_classification_apple import RandomForest

        forest = RandomForest(forestConfigStr)
        tsd = loadTSD(filename, force_update=True)
        activityType = tsd.reportedActivityType

        return LabeledFeatureSet.fromTSD(tsd, forest)

def getFeatureSetsFromTrustedEventsPickle(platform, config):
    if platform == 'android':
        from rr_mode_classification_opencv import RandomForest
    elif platform == 'ios':
        from rr_mode_classification_apple import RandomForest

    forest = RandomForest(config)
    with open('./trusted_tsds.pickle') as f:
        tsds = pickle.load(f)

    fsets = []
    for tsd in tqdm(tsds):
        fset = LabeledFeatureSet.fromTSD(tsd, forest, event_indices=tsd.trustedEventIndices)
        fset.forestConfigStr = config
        fset.forestConfigHash = sha256_sorted_json(json.loads(config))
        fsets.append(fset)

    return fsets

FSET_FILE_CACHE = {}
def loadFeatureSets(filename):
    if filename not in FSET_FILE_CACHE:
        with open(filename) as f:
            FSET_FILE_CACHE[filename] = pickle.load(f)
    return FSET_FILE_CACHE[filename]

def loadAllFeatureSets(platform, seed, fraction, include_crowd_data, use_processes, exclude_labels, do_print=True):
    all_sets = []

    filenames = glob.glob('./data/*.fsets.{}.pickle'.format(platform))
    if fraction < 1.0:
        random.seed(seed)
        print "Training on a random subset of feature sets: {:.2f}".format(fraction)
        filenames = random.sample(filenames, max(1, int(len(filenames) * fraction)))

    with Timer() as t:
        for filename in filenames:
            sets = loadFeatureSets(filename)
            if len(sets) > 0 and sets[0].reportedActivityType not in exclude_labels:
                all_sets += sets

    print "Loaded {} labeled feature sets from {} files in {:.1f}s".format(len(all_sets), len(filenames), t.elapsed)

    if include_crowd_data:
        config = all_sets[0].forestConfigStr

        print "Loading whitelisted TSDs"
        all_sets += getFeatureSetsFromAllTrainableTSDs(platform, config, use_processes=use_processes)

        print "Loading trusted TSD events"
        all_sets += getFeatureSetsFromTrustedEventsPickle(platform, config)

    return all_sets

def buildModelFromFeatureSetPickles(output_dir, split, exclude_labels,
        use_processes=True,
        fraction=1.0,
        include_crowd_data=False,
        platform='android',
        seed=None,
        builder_kwargs={}):

    print "Building model to directory: {}".format(output_dir)

    all_sets = loadAllFeatureSets(
        platform=platform,
        seed=seed,
        fraction=fraction,
        include_crowd_data=include_crowd_data,
        use_processes=use_processes,
        exclude_labels=exclude_labels)

    return buildModelFromFeatureSets(output_dir,
        all_sets,
        split=split,
        builder_kwargs=builder_kwargs,
        seed=seed,
        platform=platform,
        include_crowd_data=include_crowd_data)

def buildModelFromFeatureSets(output_dir,
        all_sets,
        split=False,
        builder_kwargs={},
        seed=None,
        include_crowd_data=True,
        platform='android'):
    if split:
        train_sets, test_sets = split_fsets(all_sets, seed=seed)
        builder = ModelBuilder(train_sets, **builder_kwargs)
    else:
        builder = ModelBuilder(all_sets, **builder_kwargs)

    output_filename = os.path.join(output_dir, 'temp.cv')
    builtforest = builder.build(output_filename=output_filename)

    if split:
        print "Confusion matrix for TEST DATA"
        print builtforest.getConfusionMatrixStr(test_sets)
    else:
        print "Confusion matrix for TRAINING DATA (no test data)"
        print builtforest.getConfusionMatrixStr(all_sets)

    builtforest.save_to_dir(output_dir, platform=platform, split=split, include_crowd_data=include_crowd_data)
    print "Saved to '{}'".format(output_dir)
    print "config.json contents:"
    with open(os.path.join(output_dir, 'config.json')) as f:
        config = json.load(f)
        print ''.join('   '+line for line in json.dumps(config, indent=2, sort_keys=True).splitlines(True))

def loadTSD(filename, force_update=False):

    pickleFilename = '{}.pickle'.format(filename)
    try:
        pickleTime = os.path.getmtime(pickleFilename)
    except:
        pickleTime = 0

    fileTime = os.path.getmtime(filename)
    if fileTime > pickleTime or force_update:
        digest = filename_sha256_hexdigest(filename)
        with open(filename) as tsdF:
            tsd = PreparedTSD(json.load(tsdF), dataHash=digest)
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

def predictTSDFileWithFilters(configPath, filename):
    with print_exceptions():
        forest = loadForestForPrediction(configPath)
        classLabels = forest.classLabels()
        try:
            tsd = loadTSD(filename)
        except:
            import traceback
            print "Skipping TSD due to exception:"
            traceback.print_exc()
            return []

        return predictTSDObjectWithFilters(tsd, forest)

def predictTSDObjectWithFilters(tsd, forest):
    confusion_tuples = []
    reportedActivityType = tsd.reportedActivityType
    for event in tsd.events:
        # skip if original predictions aren't known
        if event.originalInferredActivityType is None:
            continue

        # skip if speed is unknown
        if event.averageSpeed is None:
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

FORESTS = {}
def loadForestForPrediction(configPath):
    if configPath not in FORESTS:
        with open(configPath) as f:
            config = json.load(f)
            platform = config['extra']['platform']
            modelPath = os.path.join(os.path.dirname(configPath), '{}.cv'.format(config['cv_sha256']))

        if platform == 'android':
            from rr_mode_classification_opencv import RandomForest
        elif platform == 'ios':
            from rr_mode_classification_apple import RandomForest


        print "Loading forest id={}".format(config['cv_sha256'])

        forest = RandomForest(configPath, modelPath)
        if not forest.canPredict():
            raise NonPredictingForestError("config at '{}' cannot predict!".format(configPath))
        FORESTS[configPath] = forest

    return FORESTS[configPath]

def printConfusion(confusion):
    print "reported label on left; prediction on top"
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

def loadModelAndTestAgainstTSDs(configPath, fraction=1.0, seed=None):
    import glob
    import random
    import os
    from tqdm import tqdm
    from multiprocessing import Pool

    forest = loadForestForPrediction(configPath)

    print "Testing model: {} (id={})".format(os.path.abspath(configPath), forest.model_hash)
    classLabels = forest.classLabels()

    with open(configPath) as f:
        config = json.load(f)

    fresh_confusion = {}
    old_confusion = {}
    with Timer() as t:
        with terminatingPool() as pool:
            tsd_files = glob.glob('./data/tsd*.jsonl')

            # exclude trusted TSDs if they were used for training

            if config['extra'].get('include_crowd_data', True):
                with open(os.path.join(os.path.dirname(__file__), 'trusted_tsds.pickle')) as f:
                    pks = list(tsd.trip_pk for tsd in pickle.load(f))
                exclude = set()
                for pk in pks:
                    for filename in tsd_files:
                        if pk in filename:
                            exclude.add(filename)
                tsd_files = [fname for fname in tsd_files if fname not in exclude]
                print "Excluded {} trusted TSDs from testing".format(len(exclude))

            if fraction < 1.0:
                print "Sampling TSDs: running predictions on {:.1%} of {} total, seed={}".format(fraction, len(tsd_files), seed)
                random.seed(seed)
                tsd_files = random.sample(tsd_files, int(len(tsd_files)*fraction))



            # big files first, for better parallelization
            sorted_files = sorted(tsd_files, key=lambda filename: os.path.getsize(filename), reverse=True)

            prediction_count = 0
            for predictions in tqdm(pool.imap_unordered(partial(predictTSDFileWithFilters, configPath), sorted_files), total=len(sorted_files)):
                for prediction in predictions:
                    try:
                        reported_type, old_type, fresh_type = prediction
                    except TypeError:

                        continue

                    prediction_count += 1

                    confusion_key = (reported_type, fresh_type)
                    fresh_confusion.setdefault(confusion_key, 0)
                    fresh_confusion[confusion_key] += 1

                    confusion_key = (reported_type, old_type)
                    old_confusion.setdefault(confusion_key, 0)
                    old_confusion[confusion_key] += 1

    print "Got {} predictions from {} TSDs in {:.1f}s".format(prediction_count, len(tsd_files), t.elapsed)

    tsd_test_results = {
        'tsds': [
            { 'sha256': filename_sha256_hexdigest(filename), 'basename': os.path.basename(filename) }
            for filename in tsd_files
        ],
        'original_confusion': old_confusion,
        'fresh_confusion': fresh_confusion,
    }
    results_filename = os.path.join(os.path.dirname(configPath), 'tsd_test_results.pickle')
    with open(results_filename, 'wb') as f:
        pickle.dump(tsd_test_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print "Saved results to {}".format(results_filename)

    print "Original prediction confusion:"
    printConfusion(old_confusion)

    print "New model confusion:"
    printConfusion(fresh_confusion)

def dispatchCommand(command, options):
    builder_kwargs = {}
    for k in ('sample_count_multiple', 'active_var_count', 'max_tree_count', 'epsilon'):
        if getattr(options, k) is not None:
            builder_kwargs[k] = getattr(options, k)

    try:
        with open(options.config_filename) as f:
            config_json = json.load(f)
    except:
        # No working configuration file; create default
        config_json = { 'sampling': { 'sample_count': options.sample_count, 'sampling_rate_hz': options.sampling_rate_hz }}
        info = inspect.getargspec(ModelBuilder.__init__)
        config_json['builder'] = { k: v for k, v in izip(info.args[-len(info.defaults):], info.defaults) }
        if options.build_to_production:
            # Production models go in ActivityPredictor/<platform>/config.json
            predictor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ActivityPredictor'))
            forest_dir = os.path.join(predictor_dir, 'models', options.platform)
        else:
            forest_dir = os.path.join(os.path.dirname(__file__), 'models', options.platform)
            try: os.makedirs(forest_dir)
            except: pass

        if options.config_filename is None:
            # write new forest to this location
            options.config_filename = os.path.join(forest_dir, 'config.json')
    else:
        # We loaded a model config
        config_json.setdefault('builder', {})

    # Whether we loaded a file or not, we should override with builder args
    # provided directly
    config_json['builder'].update(builder_kwargs)

    print "Loaded configuration: "
    print ''.join('   '+line for line in json.dumps(config_json, indent=2).splitlines(True))

    np.seterr(all='raise')

    if options.sample_fraction is not None:
        fraction = options.sample_fraction
    else:
        fraction = 1.0

    if command == 'updateSamples':
        updateSamplePickles(force_update=options.force_update)
    elif command == 'updateFeatures':
        updateFeatureSets(force_update=options.force_update, platform=options.platform, config=json.dumps(config_json))
    elif command == 'train':

        try:
            exclude_labels = [int(s) for s in options.exclude_labels.split(',')]
        except:
            exclude_labels = []

        return buildModelFromFeatureSetPickles(
            output_dir=os.path.dirname(options.config_filename),
            use_processes=options.use_processes,
            split=options.split,
            fraction=fraction,
            exclude_labels=exclude_labels,
            include_crowd_data=not options.exclude_crowd_data,
            platform=options.platform,
            seed=options.seed,
            builder_kwargs=config_json['builder'])
    elif command == 'test':
        loadModelAndTestAgainstTSDs(options.config_filename, fraction=fraction, seed=options.seed)
    elif command == 'all':
        dispatchCommand('updateSamples', options)
        dispatchCommand('updateFeatures', options)
        dispatchCommand('train', options)

    else:
        raise ValueError("Unknown command: {}".format(command))

def getPipelineParser():
    import argparse
    parser = argparse.ArgumentParser(description="run a pipeline command")
    parser.add_argument('command', metavar='CMD', type=str)
    parser.add_argument('-p', '--platform', dest='platform', required=True, choices=['ios', 'android'])
    parser.add_argument('-f', '--force', dest='force_update', action='store_true', default=False)
    parser.add_argument('--exclude-labels', dest='exclude_labels', type=str, default='9')
    parser.add_argument('--no-split', dest='split', default=True, action='store_false')
    parser.add_argument('-c', '--config', dest='config_filename', metavar='CONFIG.JSON', default=None)
    parser.add_argument('-s', '--seed', dest='seed', default=None)
    parser.add_argument('--sample-fraction', dest='sample_fraction', default=None, type=float)
    parser.add_argument('--sample-count', dest='sample_count', default=64, type=int)
    parser.add_argument('--sampling-rate-hz', dest='sampling_rate_hz', default=21., type=float)
    parser.add_argument('--use-threads', dest='use_processes', default=True, action='store_false')

    parser.add_argument('-P', '--production', dest='build_to_production', default=False, action='store_true')

    parser.add_argument('--train-sample-count-multiple', dest='sample_count_multiple', default=None, type=float)
    parser.add_argument('--train-active-var-count', dest='active_var_count', default=None, type=int)
    parser.add_argument('--train-max-tree-count', dest='max_tree_count', default=None, type=int)
    parser.add_argument('--train-epsilon', dest='epsilon', default=None, type=float)

    parser.add_argument('--exclude-crowd-data', dest='exclude_crowd_data', action='store_true', default=False)

    return parser


if __name__ == '__main__':

    args = getPipelineParser().parse_args()
    try:
        dispatchCommand(args.command, args)
    except ValueError as e:
        if str(e).startswith('Unknown command'):
            print str(e)
        else:
            raise
