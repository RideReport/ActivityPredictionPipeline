# Getting started


Create a virtualenv with support for globally installed Python packages, then
install Python requirements:

```
virtualenv v
. v/bin/activate
./compile-pip
```

Next, install our fork of opencv with Python support:

```
git submodule update --init
./build_opencv.sh
```

(Be sure to run this with the virtualenv enabled)

Now you're set up and ready to collect samples, train, and test models!

# Basic Workflow

To go from sensor data collected by Motion app to a build model, just run the
`all` command with appropriate flags:

```
$ python pipeline.py all -p android --no-split -f
$ python pipeline.py all -p ios --no-split -f
```

Models are then present in `./models/android` and `./models/ios`.

You can also do each step individually; for example, to build a new Android model:

```
$ python pipeline.py updateSamples -p android --no-split -f
$ python pipeline.py updateFeatures -p android --no-split -f
$ python pipeline.py train -p android --no-split -f
```

Each step is explained in more detail below.

## Initial processing & filtering with `updateSamples`

Data obtained from the "Motion" app is called "classification data" and is
stored in JSON files in the `data` directory.  For dumb reasons, they're called
`data/classification_data.<integer>.jsonl`:

```
$ ls -l data/classification_data.*.jsonl
-rw-rw-r-- 1 evan evan   221565 Mar 15 13:41 data/classification_data.14.jsonl
-rw-rw-r-- 1 evan evan  1053042 Mar 15 13:41 data/classification_data.15.jsonl
-rw-rw-r-- 1 evan evan  4846198 Mar 15 13:41 data/classification_data.16.jsonl
...
```

These are raw traces produced by the Motion app. The `updateSamples` command
processes these files, producing for each a derivative called, for example
`[...].16.jsonl.ciSamples.pickle`. This file contains an array of
`ContinuousIntervalSample` objects. Each object contains:

 * a list of accelerometer readings at sufficient frequency with no gaps,
 * some speed data aligned to the same time axis, and
 * some metadata about the sample.

Each sample object is limited to a maximum length of `MAX_INTERVAL_LENGTH`, 60
seconds at this writing.

The default behavior of `updateSamples` is to only process new data (based on
file modification times). If the `-f` option ("force") is specified, each input
file is processed without exception.

## Computing features with `updateFeatures`

In the `updateFeatures` command, each `ContinuousIntervalSample` is transformed
to a `LabeledFeatureSet`. We use a rolling window and the feature creation
function `prepareFeaturesFromSignal` provided by our C++ library.

The feature creation function varies slightly between platforms, so this step
needs to be run once for each platform. (due to alternate FFT implementations)

The resulting object is saved to a location called, for example,
`[...].ciSamples.pickle.fsets.android.pickle`.

The default behavior of `updateFeatures` is to only process new data (based on
file modification times). If the `-f` option ("force") is specified, each input
file is processed without exception.

### Feature prep parameters

Two parameters are used for feature computation:

 * `--sample_count`: Number of samples required; must be a power of 2 for FFT features. Default: `64`
 * `--sampling_rate_hz`: Frequency of samples, in samples per second. Default: `21`

If a config file is specified with `-c <config.json>`, values from that config
file will be used instead of the command line options.

### Workflow for modifying `prepareFeaturesFromSignal`

After modifying C++ code, you just need to rebuild the shared library and then
re-run `updateFeatures -f`:

```
make
python pipeline.py updateFeatures -f
```

## Training with `train`

The training step uses labeled features to train a random forest.

*Destination*: Set the output directory and set default configuration parameters
by specifying `--config <.../config.json>`. Model build

*Splitting*: If you are testing a forest, you can split the feature sets into
train & test chunks. This is the default behavior;  the splitting is random by
default but you can control the split with the `-s --seed` option. To disable
splitting for a production model, use `--no-split`.

*Using a subset of data*: You can exclude certain types of motion by supplying a
comma-separated list of labels to the `--exclude-labels` options. For example,
use `--exclude-labels 1,9` to exclude running and my dumb tram data. The default
excludes label 9.

*Don't use TSD samples*: TSD samples are promising but can produce bad models.
I recommend specifying `--exclude-crowd-data` until we get better at
pre-processing TSDs.

*Fast iteration*: Sometimes you want the training to finish faster at the
expense of prediction accuracy. Use `--sample-fraction` to specify a value
between 0 and 1 to train on a random subset of the feature sets. Use
`-s --seed` to keep the subset the same between runs, if that matters to you.

### Training parameters:

 * `--train-sample-count-multiple`: Minimum number of samples in a decision
   node, expressed as a fraction of total available samples. Default: `0.0005`
 * `--train-active-var-count`: Number of variables that can be used for a
   decision node. The value `0` means "square root of the number of features."
   Default: `0`.
 * `--train-max-tree-count`: Maximum number of trees in final model. The primary
   termination criteria. Default: `10`
 * `--train-epsilon`: Another termination criteria. Default: `0.0001`.

# Command-line options

 * `-p --platform`: Name of target platform, `ios` or `android`. Affects `updateFeatures` and `train` and `test`.
 * `-f --force`: Force updating derivatives
 * `--exclude-labels <1,2,3>`: Exclude specified classes of motion from training, used only in `train` command.
 * `--no-split`: Disable default train/test split
 * `-c --config`: Load configuration parameters from specified file. Overrides command-line parameters.
 * `-s --seed <value>`: Seed for random number generator used in splitting or in `--sample-fraction`. Any string value is accepted.
 * `--sample-fraction`: Train on a fraction of available samples.
 * `--sample-count`: Number of readings used in feature creation. Affects `updateFeatures` and `test`.
 * `--sampling-rate-hz`: Frequency of accelerometer readings. Training data will be resampled to this frequency.
 * `--use-threads`: Parallelize with threads. Not generally recommended; the default uses processes which is faster for most things.
 * `-P --production`: Build to the production model location
 * `--train-sample-count-multiple`: See `train`.
 * `--train-active-var-count`: See `train`.
 * `--train-max-tree-count`: See `train`.
 * `--train-epsilon`: See `train`.
 * `--exclude-crowd-data`: Disable TSDs during training. Recommended.
