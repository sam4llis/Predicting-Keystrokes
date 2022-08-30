from make_model import KeyClusters
from make_model import TDOA
from make_model import CrossPredict
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sn
import joblib

# GLOBAL VARIABLES
KEYS = {
        'a' : 0,
        'b' : 1,
        'c' : 2,
        'd' : 3,
        'e' : 4,
        'f' : 5,
        'g' : 6,
        'h' : 7,
        'i' : 8,
        'j' : 9,
        'k' : 10,
        'l' : 11,
        'm' : 12,
        'n' : 13,
        'o' : 14,
        'p' : 15,
        'q' : 16,
        'r' : 17,
        's' : 18,
        't' : 19,
        'u' : 20,
        'v' : 21,
        'w' : 22,
        'x' : 23,
        'y' : 24,
        'z' : 25,
        '0' : 26,
        '1' : 27,
        '2' : 28,
        '3' : 29,
        '4' : 30,
        '5' : 31,
        '6' : 32,
        '7' : 33,
        '8' : 34,
        '9' : 35,
        'Key.enter' : 36,
        'Key.esc' : 37,
        'Key.backspace' : 38,
        'Key.space' : 39
       }


class PredictFromModel:

    def __init__(self, model, clust, file_directory, filenames=None):
        self.model = model
        self.clust = clust
        self._load_data(file_directory, filenames)

    def _load_data(self, file_directory, filenames):
        global KEYS

        self.X_test, self.Y_test = [], []
        if not filenames:
            filepaths = self._get_filepaths(file_directory)
        else:
            filepaths = [os.path.join(file_directory, filename) for filename in filenames]

        count = 0
        self.tau = []
        for filepath in filepaths:
            _, filename = os.path.split(filepath)
            key_id = self._get_key_id(filename)

            for key, val in KEYS.items():
                if key == key_id:
                    tdoa = TDOA(filepath, cc_algorithm='cc')
                    data = np.append(self._get_mfcc_data(filepath), tdoa.tau)
                    self.X_test.append(data)
                    self.Y_test.append(key_id)
                    count += 1
                    break
        print(f'{count}/{len(filepaths)} files processed.')

        scaler = StandardScaler()
        scaler.n_features_in_ = len(self.X_test[0])
        scaler.scale_ = np.array(joblib.load('code/scaler_scale.bin'))
        scaler.mean_ = np.array(joblib.load('code/scaler_mean.bin'))
        scaler.var_ = np.array(joblib.load('code/scaler_var.bin'))

        # scaler = joblib.load('code/std_scaler.bin')
        self.X_test = scaler.transform(self.X_test)
        # self._get_clusters()
        self._predict_clusters()
        score = self.model.score(self.X_test, self.Y_test)
        print(score*100)

    def to_probability_dict(self):
        d = []
        keys = self.model.classes_
        for i in range(len(self.X_test)):
            vals = self.model.predict_proba(self.X_test)[i]
            d.append(dict(zip(keys, vals)))
        return d


    def _get_filepaths(self, file_directory):
        filepaths = glob.iglob(os.path.join(file_directory, '**', '*.wav'), recursive=True)
        return [filepath for filepath in filepaths]

    def _get_key_id(self, filename):
        f, _ = os.path.splitext(filename)
        s = ''.join([i for i in f if not i.isdigit()])
        # Case of digit:
        if s == '':
            s = f[0]
        return s

    def _get_mfcc_data(self, filepath):
        y, sr = librosa.load(filepath, sr=96000, duration=0.30)
        y = librosa.util.normalize(y)

        y_mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13, hop_length=int(0.01*sr), fmax=14000, fmin=400)
        y_delta_mfcc = librosa.feature.delta(y_mfcc, width=7 , mode='nearest')
        y_delta2_mfcc = librosa.feature.delta(y_mfcc, width=7 , order=2 , mode='nearest')

        y_mfcc = [np.mean(v) for v in y_mfcc]
        # return y_mfcc
        y_delta_mfcc = [np.mean(v) for v in y_delta_mfcc]
        y_delta2_mfcc = [np.mean(v) for v in y_delta2_mfcc]
        return np.concatenate((y_mfcc, y_delta_mfcc, y_delta2_mfcc))

    def _predict_clusters(self):
        tau = [entry[-1] for entry in self.X_test]
        X_test = np.array(list(zip(tau, [0] * len(tau))))
        test_labels = self.clust.predict(X_test)
        self.X_test = np.delete(self.X_test, (-1), axis=1)
        self.X_test = np.c_[self.X_test, test_labels]


    def _get_clusters(self, n_clusters=15):
        tau_train = [entry[-1] for entry in self.X_train] # get only tau measurement
        X_train = np.array(list(zip(tau_train, [0] * len(tau_train)))) #ONLY X_train HERE! may not need 0 here?
        kmeans = KMeans(n_clusters=n_clusters).fit(X_train)# FIT USING X_train, PREDICT using X_test!
        train_labels = kmeans.labels_

        self.X_train = np.delete(self.X_train, (-1), axis=1) # drop tdoa col
        self.X_train = np.c_[self.X_train, train_labels]

        tau_test = [entry[-1] for entry in self.X_test]
        X_test = np.array(list(zip(tau_test, [0] * len(tau_test))))
        test_labels = kmeans.predict(X_test)

        self.X_test = np.delete(self.X_test, (-1), axis=1)
        self.X_test = np.c_[self.X_test, test_labels]

    def _get_tdoa(self, filepath):
        tdoa = TDOA(filepath)
        return tdoa.tau


# def main():
#     file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface_t', 'Shure SM7B')
#     model_name = os.path.join('code', 'finalised_model_sm7b.pkl')
#     model_sm7b_class = joblib.load(model_name)

#     model_name = os.path.join('code', 'finalised_kmeans_sm7b.pkl')
#     model_sm7b_clust = joblib.load(model_name)

#     model_sm7b_class

def main():

    file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface_t', 'Shure SM7B')

    model_name = os.path.join('code', 'finalised_model_sm7b.pkl')
    model_sm7b_class = joblib.load(model_name)

    model_name = os.path.join('code', 'finalised_kmeans_sm7b.pkl')
    model_sm7b_clust = joblib.load(model_name)

    keys_sm7b = PredictFromModel(model_sm7b_class, model_sm7b_clust, file_directory)
    d_sm7b = keys_sm7b.to_probability_dict()

    file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface_t', 'Rode NT1A')

    model_name = os.path.join('code', 'finalised_model_nt1a.pkl')
    model_nt1a_class = joblib.load(model_name)

    model_name = os.path.join('code', 'finalised_kmeans_nt1a.pkl')
    model_nt1a_clust = joblib.load(model_name)

    keys_nt1a = PredictFromModel(model_nt1a_class, model_nt1a_clust, file_directory)
    d_nt1a = keys_nt1a.to_probability_dict()

    cross_predict = CrossPredict(d_sm7b, d_nt1a, keys_sm7b.Y_test)
    cross_predict.show_accuracy(confmatrix=False)
    print(cross_predict.ans)

if __name__ == "__main__":
    main()



