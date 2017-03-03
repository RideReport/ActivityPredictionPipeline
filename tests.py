import unittest
import numpy as np
from scipy import interpolate
import os
import json
import platform
from itertools import izip
import dateutil.parser
from operator import itemgetter

import math
from pipeline import AccelerationVector3D

import pytz
import datetime

from contexttimer import Timer

# Boost wrappers
from utilityadapter import UtilityAdapter
from rr_mode_classification_opencv import RandomForest as OpenCVRandomForest
from opencv_fft import OpenCVFFTPythonAdapter
if platform.system() == 'Darwin':
    from apple_fft import AppleFFTPythonAdapter
    from rr_mode_classification_apple import RandomForest as AppleRandomForest

# other wrappers
import shlex
import subprocess
import json

class JSONCommandProcess(object):
    def __init__(self, cmd_string, capture_stderr=True):
        self.capture_stderr = capture_stderr
        args = shlex.split(cmd_string)
        kwargs = {
            'stdin': subprocess.PIPE,
            'stdout': subprocess.PIPE,
        }
        if self.capture_stderr:
            kwargs['stderr'] = open(os.devnull, 'w')
        self.pipe = subprocess.Popen(args, **kwargs)

        # Wait for ready signal from app
        line = self.pipe.stdout.readline()
        assert json.loads(line)['ready']

    def call(self, method, **kwargs):
        kwargs['method'] = method
        command = json.dumps(kwargs)
        self.pipe.stdin.write('{}\n'.format(command))
        try:
            line = self.pipe.stdout.readline()
            return json.loads(line)
        except ValueError as e:
            # print "command: {}".format(command)
            print "output: {}".format(line)
            raise e




import logging
logger = logging.getLogger(__name__)

owndir = os.path.dirname(__file__)



class FixtureDict(object):

    def __init__(self, dirname):
        self._fixture_data = {}
        self._dirname = dirname

    def __getitem__(self, k):
        if k not in self._fixture_data:
            self._fixture_data[k] = self._load(k)
        return self._fixture_data[k]

    def _load(self, k):
        for filename_pattern in ('{}.json', '{}'):
            filename = filename_pattern.format(k)
            try:
                with open(os.path.join(self._dirname, filename))  as f:
                    return json.load(f)
            except (IOError, ValueError) as exc:
                raise
                logger.info('Failed to load "{}": {}'.format(filename, repr(exc)))

        raise ValueError('Could not find fixture "{}"'.format(k))

class FixtureMixin(object):
    data = FixtureDict(os.path.join(os.path.dirname(__file__), 'fixtures'))

class TestResample(unittest.TestCase):

    def test_resample(self):
        from training_tools import resampleLine
        line = {}
        line['rawNormsSeconds'] = np.arange(0, 30, 1, dtype=float)
        line['rawNorms'] = np.sin((line['rawNormsSeconds'] * 2. / np.pi), dtype=float)
        originalNorms = line['rawNorms']
        resampleLine(line, 2.)
        # Should be spaced half-second apart
        self.assertEqual(0.5, np.median(np.diff(line['rawNormsSeconds'])))

        # Every other element should be the same as the original elements
        # (after slicing out the interior, like continuousSecondsIndices()
        # does)
        self.assertTrue(np.all(originalNorms[10:20] == line['rawNorms'][0::2]))

class TestInterpolation(unittest.TestCase, FixtureMixin):
    def setUp(self):
        pass

    def np_interpolateLinearRegular(self, inputX, inputY, newSpacing, outputLength):
        newX = np.arange(inputX[0], inputX[-1], newSpacing)
        outputY = np.interp(newX, inputX, inputY)
        if len(outputY) < outputLength:
            raise ValueError("Output is not long enough")
        return list(outputY[:outputLength])

    def np_interpolateSplineRegular(self, inputX, inputY, newSpacing, outputLength):
        newX = np.arange(inputX[0], inputX[-1], newSpacing)
        s = interpolate.InterpolatedUnivariateSpline(inputX, inputY)
        newY = s(newX)
        if len(newY) < outputLength:
            raise ValueError("Output is not long enough")
        return list(newY[:outputLength])

    def np_interpolateCubicRegular(self, inputX, inputY, newSpacing, outputLength):
        newX = np.arange(inputX[0], inputX[-1], newSpacing)
        s = interpolate.interp1d(inputX, inputY, kind='cubic')
        newY = s(newX)
        if len(newY) < outputLength:
            raise ValueError("Output is not long enough")
        return list(newY[:outputLength])

    def plot_interpolations(self, inputX, inputY, newSpacing, outputLength, **kwargs):
        import matplotlib.pyplot as plt
        newX = np.arange(inputX[0], inputX[-1], newSpacing)[:outputLength]
        plt.figure()
        plt.plot(inputX, inputY, 'x-', label='input', alpha=0.2)
        for k, v in kwargs.items():
            plt.plot(newX, v, label=k, alpha=0.5)
        plt.legend()
        plt.show()

    def test_interpolator_fails_for_too_little_data(self):
        utilityAdapter = UtilityAdapter()

        with self.assertRaises(RuntimeError):
            cInterpolated = utilityAdapter.interpolateSplineRegular([0, 1, 2], [1, 1, 1], 1./20., 1000)

    def test_spline_congruent(self):
        "These differ a little bit more than linear, perhaps because alg uses doubles internally"
        utilityAdapter = UtilityAdapter()
        accVec = AccelerationVector3D(self.data['androidAccelerations'])
        spacing = 1./20.
        samples = 64;
        npInterpolated = self.np_interpolateSplineRegular(accVec.seconds, accVec.norms, spacing, samples)
        cInterpolated = utilityAdapter.interpolateSplineRegular(accVec.seconds, accVec.norms, spacing, samples)
        # self.plot_interpolations(accVec.seconds, accVec.norms, spacing, samples, np=npInterpolated, c=cInterpolated)

        self.assertEqual(len(npInterpolated), len(cInterpolated))
        try:
            for npValue, cValue in izip(npInterpolated, cInterpolated):
                self.assertAlmostEqual(npValue, cValue, delta=5e-6)
        except AssertionError:
            print "np: {}".format(npInterpolated)
            print "c : {}".format(cInterpolated)
            raise

class TestForest(unittest.TestCase, FixtureMixin):

    def setUp(self):
        self.sampleSize = 64
        self.samplingRateHz = 20
        self.modelFilename = os.path.join(os.path.dirname(__file__), 'testingForest.cv')

    def _test_deterministic(self, forest, fixtureName):
        utilityAdapter = UtilityAdapter()
        accVec = AccelerationVector3D(self.data[fixtureName])
        prev_confidences = None
        for i in xrange(0, 100):
            confidences = forest.classifySignal(accVec.readings)
            if prev_confidences is not None:
                self.assertEqual(confidences, prev_confidences)

            prev_confidences = confidences

    def test_opencv_deterministic(self):
        forest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)
        self._test_deterministic(forest, 'androidAccelerations')

    def _offset_resample_results(self, forest, accVec, secondsOffset):
        """Use `forest` and data in `fixtureName` to compute predictions with a time offset

        Returns { label: confidence } predictions dict
        """
        features = forest.prepareFeaturesFromSignal(accVec.readings, secondsOffset)

        confidences = forest.classifyFeatures(features)
        predictions = dict(izip(forest.classLabels(), confidences))
        return predictions

    def test_offset_resample_results_opencv(self):
        forest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)
        prev_ordered = None

        accVec = AccelerationVector3D(self.data['androidAccelerations'])
        for offset in np.arange(0, 14e-3, 1e-4):
            predictions = self._offset_resample_results(forest, accVec, offset)
            nonzero_predictions = { k: v for k, v in predictions.items() if predictions[k] > 0 }
            ordered = sorted(nonzero_predictions.keys(), key=lambda k: nonzero_predictions[k], reverse=True)
            if prev_ordered is not None:
                self.assertEqual(prev_ordered, ordered)
            prev_ordered = ordered

    @unittest.skipIf(platform.system() != 'Darwin', 'Requires Apple system')
    def test_python_compare_rf_features(self):
        self.assertNotEqual(AppleRandomForest, OpenCVRandomForest)
        appleForest = AppleRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)
        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)

        accVec = AccelerationVector3D(self.data['androidAccelerations'])

        appleFeatures = appleForest.prepareFeaturesFromSignal(accVec.readings)
        opencvFeatures = opencvForest.prepareFeaturesFromSignal(accVec.readings)

        self.assertEqual(len(appleFeatures), len(opencvFeatures))

        for index, (appleF, opencvF) in enumerate(izip(appleFeatures, opencvFeatures)):
            self.assertAlmostEqual(appleF, opencvF, msg='feature {} should be equal'.format(index), places=6)

    def test_opencv_classify_vs_features(self):
        accVec = AccelerationVector3D(self.data['androidAccelerations'])

        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)
        features = opencvForest.prepareFeaturesFromSignal(accVec.readings)
        confidences = opencvForest.classifyFeatures(features)

        directConfidences = opencvForest.classifySignal(accVec.readings)
        self.assertEqual(confidences, directConfidences)

        reversedDirect = opencvForest.classifySignal(list(reversed(accVec.readings)))
        self.assertEqual(directConfidences, reversedDirect)

    def test_java_vs_python_classify_signal(self):
        javaProcess = JSONCommandProcess('java -jar mode_classification_wrapper/java/build/libs/java-all.jar "{}"'.format(self.modelFilename), capture_stderr=True)
        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)
        accVec = AccelerationVector3D(self.data['androidAccelerations'])

        javaPredictions = javaProcess.call('classifyAccelerometerSignal', readings=accVec.readings)
        if javaPredictions.get('error'):
            self.fail(javaPredictions['detail'])

        confidences = opencvForest.classifySignal(accVec.readings)
        opencvPredictions = { str(k): v for k, v in izip(opencvForest.classLabels(), confidences) }
        self.assertEqual(javaPredictions, opencvPredictions)

    def test_tsds_agree(self):
        javaProcess = JSONCommandProcess('java -jar mode_classification_wrapper/java/build/libs/java-all.jar "{}"'.format(self.modelFilename), capture_stderr=True)
        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)

        self._test_tsd_predictions_agree_using_av3(self.data['tsd-3e82d1a8-1d19-46cf-9e66-fa630d6892f8'], javaProcess, opencvForest)

    def test_tsds_agree_using_process(self):
        javaProcess = JSONCommandProcess('java -jar mode_classification_wrapper/java/build/libs/java-all.jar "{}"'.format(self.modelFilename), capture_stderr=True)
        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)

        self._test_tsd_predictions_agree_using_process(self.data['tsd-3e82d1a8-1d19-46cf-9e66-fa630d6892f8'], javaProcess, opencvForest)

    def _test_tsd_predictions_agree_using_av3(self, tsd, process, forest):
        for index, sdc in enumerate(tsd['data']):
            accVec = AccelerationVector3D(sdc['accelerometerAccelerations'])
            Ppredictions = process.call('classifyAccelerometerSignal', readings=accVec.readings)
            if Ppredictions.get('error'):
                self.fail(Ppredictions['detail'])
            Fconfidences = forest.classifySignal(accVec.readings)
            Fpredictions = { str(k): v for k, v in izip(forest.classLabels(), Fconfidences) }
            self.assertEqual(Ppredictions, Fpredictions, "confidences differ for tsd__trip={} sdc_index={}, P={} F={}".format(tsd['trip_pk'], index, Ppredictions, Fpredictions))

    def _test_tsd_predictions_agree_using_process(self, tsd, process, forest):
        output = process.call('classifyTSD', tsd=tsd)
        if output.get('error'):
            self.fail(output['detail'])

        all_Ppredictions = output['predictions']

        for index, sdc in enumerate(tsd['data']):
            accVec = AccelerationVector3D(sdc['accelerometerAccelerations'])
            Fconfidences = forest.classifySignal(accVec.readings)
            Fpredictions = { str(k): v for k, v in izip(forest.classLabels(), Fconfidences) }
            Ppredictions = all_Ppredictions[index]
            self.assertEqual(Ppredictions, Fpredictions, "confidences differ for tsd__trip={} sdc_index={}, P={} F={}".format(tsd['trip_pk'], index, Ppredictions, Fpredictions))

@unittest.skipIf(os.getenv('TEST_MODEL_FILE') is None, "Run with TEST_MODEL_FILE=/path/to/forest.cv to test an arbitrary forest")
class TestEnvironmentForest(TestForest):
    def setUp(self):
        super(TestEnvironmentForest, self).setUp()
        self.modelFilename = os.getenv('TEST_MODEL_FILE')

    @unittest.skipIf(os.getenv('TEST_TSD_GLOB') is None, "Run with TEST_MODEL_FILE=/path/to/forest.cv TEST_TSD_GLOB='data/*.tsd.json' to test against TSDs")
    def test_tsds_list(self):
        from tqdm import tqdm
        import glob
        javaProcess = JSONCommandProcess('java -jar mode_classification_wrapper/java/build/libs/java-all.jar "{}"'.format(self.modelFilename), capture_stderr=True)
        opencvForest = OpenCVRandomForest(self.sampleSize, self.samplingRateHz, self.modelFilename)

        failed = {}
        succeeded = {}
        for tsd_filename in tqdm(glob.glob(os.getenv('TEST_TSD_GLOB'))):
            with open(tsd_filename) as f:
                try:
                    tsd = json.load(f)
                except ValueError:
                    pass
                    # print "test_tsds_list: skipping file '{}'".format(tsd_filename)
                else:
                    # print "test_tsds_list: opened file '{}'".format(tsd_filename)
                    try:
                        with Timer() as t:
                            self._test_tsd_predictions_agree_using_process(tsd, javaProcess, opencvForest)
                        succeeded[tsd_filename] = True
                        # print "test_tsds_list: finished tsd for '{}' in {: 4.1f}s".format(tsd_filename, t.elapsed)
                    except Exception as e:
                        failed[tsd_filename] = repr(e)
                        # print "test_tsds_list: failed tsd '{}': {}".format(tsd_filename, repr(e))

        print "test_tsds_list: {} good, {} bad".format(len(succeeded), len(failed))
        exceptions = {}
        for filename, exc_str in failed.iteritems():
            exceptions[exc_str].setdefault([])
            exceptions[exc_str].append(filename)

        for exc_str, names in exceptions.iteritems():
            print "test_tsds_list: count={}: {}".format(len(names), exc_str)


class TestFFT(unittest.TestCase, FixtureMixin):

    @unittest.skipIf(platform.system() != 'Darwin', 'Requires Apple system')
    def test_compare_fft_results(self):
        sampleSize = 64
        appleFFT = AppleFFTPythonAdapter(sampleSize)
        opencvFFT = OpenCVFFTPythonAdapter(sampleSize)

        accVec = AccelerationVector3D(self.data['androidAccelerations'])
        norms = accVec.norms
        try:
            for index, (appleValue, opencvValue) in enumerate(izip(appleFFT.fft(norms), opencvFFT.fft(norms))):
                if index < sampleSize / 2:
                    fractional_difference = (appleValue - opencvValue) / opencvValue
                    self.assertLess(abs(fractional_difference), 1e-4)
        except AssertionError:
            print "Mismatch at index {}".format(index)
            for index,  (appleValue, opencvValue) in enumerate(izip(appleFFT.fft(norms), opencvFFT.fft(norms))):
                print "i: {: 2} apple: {: 12.6f} opencv: {: 12.6f}".format(index, appleValue, opencvValue)
            raise

if __name__ == '__main__':
    unittest.main()
