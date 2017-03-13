
import numpy as np
import sys
import os
import pipeline
from pipeline import LabeledFeatureSet, PreparedTSD
import argparse
import operator
from tqdm import tqdm

def csv_to_float_array(text):
    elements = [float(n) for n in text.split(',')]
    if len(elements) == 1:
        return np.array(elements)
    return np.arange(*elements)

def csv_to_int_array(text):
    elements = [int(n) for n in text.split(',')]
    if len(elements) == 1:
        return np.array(elements)
    return np.arange(*elements)

CACHE = {}
def getFeatureSets(**kwargs):
    key = 'fsets:{}'.format(pipeline.sha256_sorted_json(kwargs))
    if key not in CACHE:
        CACHE[key] = pipeline.loadAllFeatureSets(**kwargs)
    return CACHE[key]

class Trainer(object):
    def __init__(self, args, **training_kwargs):
        self.args = argparse.Namespace()
        self.args.__dict__ = vars(args)

        self.builder_kwargs = training_kwargs

    def run(self):
        with pipeline.print_exceptions():
            try:
                exclude_labels = [int(s) for s in self.args.exclude_labels.split(',')]
            except:
                exclude_labels = []

            # print "excluding {}".format(exclude_labels)

            fsets = getFeatureSets(
                platform=args.platform,
                seed=args.seed,
                fraction=1.0,
                include_crowd_data=not self.args.exclude_crowd_data,
                use_processes=False,
                exclude_labels=exclude_labels)

            forest_dir = os.path.join(os.path.dirname(__file__), 'models', self.args.platform)

            output_dir = os.path.join(forest_dir, pipeline.sha256_sorted_json(self.builder_kwargs))
            try: os.makedirs(output_dir)
            except: pass

            pipeline.buildModelFromFeatureSets(
                output_dir=output_dir,
                all_sets=fsets,
                split=self.args.split,
                seed=self.args.seed,
                include_crowd_data=not self.args.exclude_crowd_data,
                platform=args.platform,
                builder_kwargs=self.builder_kwargs
            )
            config_filename = os.path.join(output_dir, 'config.json')
            return config_filename

def runTrainer(trainer):
    return trainer.run()

def generateTrainers(args, sample_count_multiple, active_var_count, max_tree_count, epsilon):
    with pipeline.print_exceptions():
        for scm in sample_count_multiple:
            for avc in active_var_count:
                for mtc in max_tree_count:
                    for eps in epsilon:
                        yield Trainer(args, sample_count_multiple=scm, active_var_count=avc, max_tree_count=mtc, epsilon=eps)

def trainParallel(args):
    training_kwargs = {}
    for k, v in vars(args).iteritems():
        if k.startswith('range_'):
            training_kwargs[k.replace('range_', '')] = v

    for k, v in training_kwargs.iteritems():
        print "{}: {}".format(k, v)
    total = reduce(operator.mul, (v.shape[0] for v in training_kwargs.values()))

    if args.dry_run:
        print "Would build {} models".format(total)
        return

    dirs = []
    try:
        with pipeline.terminatingPool() as pool:
            for config_filename in tqdm(pool.imap_unordered(runTrainer, generateTrainers(args, **training_kwargs)), total=total):
                args.config_filename = config_filename
                pipeline.dispatchCommand('test', args)
                dirs.append(os.path.dirname(config_filename))
    finally:
        print "COMPLETED DIRS:"
        for d in dirs:
            print d

if __name__ == '__main__':
    parser = pipeline.getPipelineParser()

    parser.add_argument('--sample-count-multiple', dest='range_sample_count_multiple', default=np.array([.0005]), type=csv_to_float_array)
    parser.add_argument('--active-var-count', dest='range_active_var_count', default=np.array([0]), type=csv_to_int_array)
    parser.add_argument('--max-tree-count', dest='range_max_tree_count', default=np.array([10]), type=csv_to_int_array)
    parser.add_argument('--epsilon', dest='range_epsilon', default=np.array([0.0001]), type=csv_to_float_array)

    parser.add_argument('--dry-run', dest='dry_run', default=False, action='store_true')

    args = parser.parse_args(sys.argv)

    trainParallel(args)
