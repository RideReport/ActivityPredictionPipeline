import pipeline
import os
import glob
from tqdm import tqdm
import json
import pickle
from itertools import imap

class NormalizedConfusionMatrix(object):
    def __init__(self, confusion):
        self.matrix = {}
        self.totals = {}
        self.labels = set(k[0] for k in confusion.keys()) | set(k[1] for k in confusion.keys())
        for reported_label in self.labels:
            total = sum(confusion.get((reported_label, i), 0) for i in self.labels)
            self.totals[reported_label] = total
            for predicted_label in self.labels:
                k = (reported_label, predicted_label)
                self.matrix[k] = int(confusion.get((reported_label, predicted_label), 0) / float(max(total, 1)) * 100)

    def sumRowWrong(self, label):
        return sum(self.matrix[label, i] for i in self.labels if i != label)

    def sumColumnWrong(self, label):
        return sum(self.matrix[i, label] for i in self.labels if i != label)

    def __getitem__(self, k):
        return self.matrix[k]

def loadModelTestResults(model_dirname):
    with pipeline.print_exceptions():
        try:
            with open(os.path.join(model_dirname, 'tsd_test_results.pickle')) as f:
                data = pickle.load(f)
        except:
            return None

        with open(os.path.join(model_dirname, 'config.json')) as f:
            config = json.load(f)
        cv_filename = os.path.join(model_dirname, '{}.cv'.format(config['cv_sha256']))
        del data['tsds']
        data['config'] = config
        data['cv_filename'] = cv_filename
        data['MB'] = int(os.path.getsize(cv_filename) / 2**20)
        # data['shasha'] = pipeline.sha256_sorted_json(data['tsds'])
        data['dirname'] = model_dirname
        data['normalized_fresh'] = NormalizedConfusionMatrix(data['fresh_confusion'])
        data['scores'] = scores(data)
        return data

def scores(test_results):
    fc = test_results['normalized_fresh']
    return {
        'bike_correct': fc[2,2],
        'bike_predicted_others': fc.sumRowWrong(2),
        'others_predicted_bike': fc.sumColumnWrong(2),
        'motor_predicted_motor': (
            fc[3,3] + fc[3,5] + fc[3,6] +
            fc[5,3] + fc[5,5] + fc[5,6] +
            fc[6,3] + fc[6,5] + fc[6,6]
        ),
        'foot_predicted_bike': (fc[1,2] + fc[4,2]),
        'bike_predicted_foot': (fc[2,1] + fc[2,4]),
    }

def describeModel(description, test_results):
    with open(os.path.join(test_results['dirname'], 'config.json')) as f:
        config = json.load(f)
    config['cv'] = pipeline.BuiltForest._loadMetaFromCvFile(test_results['cv_filename'])
    print "{}: {}".format(description, test_results['dirname'])
    print "File size: {} MB".format(test_results['MB'])
    print "Builder parameters:"
    print json.dumps(config['builder'], indent=2)
    print "Internal CV parameters:"
    print json.dumps(config['cv']['opencv_ml_rtrees']['training_params'], indent=2)
    print "Classes: {}".format(config['cv']['opencv_ml_rtrees']['class_labels'])
    print "Ntrees: {}".format(config['cv']['opencv_ml_rtrees']['ntrees'])
    print "Original confusion:"
    pipeline.printConfusion(test_results['original_confusion'])
    print "Fresh confusion:"
    pipeline.printConfusion(test_results['fresh_confusion'])
    print "Scores:"
    print json.dumps(scores(test_results), indent=2, sort_keys=True)



def loadAllResults(dirnames):
    with pipeline.terminatingPool(True) as pool:
        print "Loading test result pickles ..."
        models = list(tqdm(pool.imap_unordered(loadModelTestResults, dirnames), total=len(dirnames)))
        return [ m for m in models if m is not None ]

def findGoodModelsAndDescribe():
    try:
        print "Trying to load aggregated data from big pickle"
        with open('./models_results.pickle') as f:
            models = pickle.load(f)
    except:
        print "Aggregated data could not be loaded;"
        models = loadAllResults(glob.glob('./models/android/*'))
        with open('./models_results.pickle', 'wb') as f:
            pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

    print "Updating scores"
    for m in tqdm(models):
        m['scores'] = scores(m)

    describeModel('Lowest bike -> other', min(models, key=lambda m: m['scores']['bike_predicted_others']))
    describeModel('Highest bike -> bike', max(models, key=lambda m: m['scores']['bike_correct']))
    describeModel('Highest motor -> motor', max(models, key=lambda m: m['scores']['motor_predicted_motor']))
    describeModel('Lowest foot -> bike', min(models, key=lambda m: m['scores']['foot_predicted_bike']))


if __name__ == '__main__':
    findGoodModelsAndDescribe()
