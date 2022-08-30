from utils import *
import numpy as np
import pandas as pd
import os
import glob
import librosa
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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
from sklearn.pipeline import make_pipeline


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

class KeyClusters:

    def __init__(self, file_directory, filenames=None):
        self._load_data(file_directory, filenames)


    def _load_data(self, file_directory, filenames):
        global KEYS

        if not filenames:
            filepaths = self._get_filepaths(file_directory)
        else:
            filepaths = [os.path.join(file_directory, filename) for filename in filenames]

        self.X, self.Y = [], []
        # filepaths = self._get_filepaths(file_directory)

        count = 0
        self.tau = []
        for filepath in filepaths:
            _, filename = os.path.split(filepath)
            key_id = self._get_key_id(filename)

            for key, val in KEYS.items():
                if key == key_id:
                    tdoa = TDOA(filepath, cc_algorithm='cc')
                    # self.tau.append(tdoa.tau)
                    # self.X.append(self._get_mfcc_data(filepath))
                    data = np.append(self._get_mfcc_data(filepath), tdoa.tau)
                    self.X.append(data)
                    self.Y.append(key_id)
                    count += 1
                    break
        print(f'{count}/{len(filepaths)} files processed.')

        scaler = StandardScaler()
        # self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, \
                test_size=0.2, random_state=200)
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        joblib.dump(scaler.scale_, 'code/scaler_scale.bin')#, compress=True)
        joblib.dump(scaler.mean_, 'code/scaler_mean.bin')#, compress=True)
        joblib.dump(scaler.var_, 'code/scaler_var.bin')#, compress=True)

        self._get_clusters()

    def ensemble_fit(self):
        # clf1 = SVC(kernel='linear', probability=True)
        # clf2 = RandomForestClassifier(n_estimators=100)
        # clf3 = SVC(kernel='rbf', probability=True)
        self.eclf = SVC(kernel='rbf', probability=True)
        # clf4 = GaussianNB()

        # self.eclf = VotingClassifier(estimators=[('svcl', clf1), ('rf', clf2), ('rbf', clf3), ('gnb', clf4)], voting='soft', weights=[1, 0, 0, 0])

        # for clf, label in zip([clf1, clf2, clf3, clf4, self.eclf], ['SVC', 'RandomForest', 'SVC RBF kernel', 'GaussianNB', 'Ensemble']):
        #     scores = cross_val_score(clf, self.X, self.Y, scoring='accuracy', cv=5)
        #     print(f'Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f}) {label}')

        # clf1 = RandomForestClassifier(n_estimators=100)
        # clf2 = GaussianNB()

        # self.eclf = VotingClassifier(estimators=[('rf', clf1), ('gnb', clf2)], voting='soft')

#         for clf, label in zip([clf1, clf2, self.eclf], ['RandomForest', 'GaussianNB', 'Ensemble']):
#             scores = cross_val_score(clf, self.X, self.Y, scoring='accuracy', cv=5)
#             print(f'Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f}) {label}')

        self.eclf.fit(self.X_train, self.y_train)
        score = self.eclf.score(self.X_test, self.y_test)
        print(score)

    def predict(self, filepath):
        predict_tdoa = TDOA(filepath, test=True)
        data = np.append(self._get_mfcc_data(filepath), predict_tdoa.tau)
        x = self.eclf.predict([data])
        # print(x)

    def to_probability_dict(self):
        d = []
        keys = self.eclf.classes_
        for i in range(len(self.X_test)):
            vals = self.eclf.predict_proba(self.X_test)[i]
            d.append(dict(zip(keys, vals)))
        return d

    def model_fit(self):
        classification_models = [
                                 # KNeighborsClassifier(),
                                 SVC(kernel='linear'),
                                 SVC(kernel='rbf'),
                                 # DecisionTreeClassifier(),
                                 RandomForestClassifier(n_estimators=100),
                                 # AdaBoostClassifier(),
                                 GaussianNB()
                                 # MLPClassifier()
                                ]
        scores = []
        clf_report = []

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, \
        #         test_size=0.2, random_state=70)

        for model in classification_models:
            model.fit(self.X_train, self.y_train)
            score = model.score(self.X_test, self.y_test)
            model_name = type(model).__name__
            if model_name == 'SVC' and model.kernel == 'rbf':
                model_name += ' RBF kernel'
            scores.append((model_name, (f'{100*score:.2f}')))
            # clf_report.append((model_name), report)
            # self.plot_confusion_matrix(model)

        scores_df = pd.DataFrame(scores, columns=['Classifier', 'Accuracy Score'])
        scores_df = scores_df.sort_values(by='Accuracy Score', axis=0, ascending=False)
        print(scores_df)

    def _get_filepaths(self, file_directory):
        # data = subset_data(file_directory, n_samples_per_class=15)
        filepaths = glob.iglob(os.path.join(file_directory, '**', '*.wav'), recursive=True)
        return [filepath for filepath in filepaths]

    def _get_key_id(self, filename):
        f, _ = os.path.splitext(filename)
        s = ''.join([i for i in f if not i.isdigit()])
        # Special case of digit.
        if s == '':
            s = f[0]
        return s

    def _get_mfcc_data(self, filepath):
        y, sr = librosa.load(filepath, sr=None, duration=0.15)
        y = librosa.util.normalize(y)

        y_mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13, hop_length=int(0.01*sr), fmax=14000, fmin=400)
        y_delta_mfcc = librosa.feature.delta(y_mfcc, width=7)
        y_delta2_mfcc = librosa.feature.delta(y_mfcc, width=7, order=2)

        y_mfcc = [np.mean(v) for v in y_mfcc]
        y_delta_mfcc = [np.mean(v) for v in y_delta_mfcc]
        y_delta2_mfcc = [np.mean(v) for v in y_delta2_mfcc]
        return np.concatenate((y_mfcc, y_delta_mfcc, y_delta2_mfcc))
        # return y_mfcc

    def _get_tdoa(self, filepath):
        tdoa = TDOA(filepath)
        return tdoa.tau

    def _get_clusters(self, n_clusters=15):
        tau_train = [entry[-1] for entry in self.X_train] # get only tau measurement
        X_train = np.array(list(zip(tau_train, [0] * len(tau_train)))) #ONLY X_train HERE! may not need 0 here?
        self.kmeans = KMeans(n_clusters=n_clusters).fit(X_train)# FIT USING X_train, PREDICT using X_test!
        train_labels = self.kmeans.labels_

        self.X_train = np.delete(self.X_train, (-1), axis=1) # drop tdoa col
        self.X_train = np.c_[self.X_train, train_labels]

        tau_test = [entry[-1] for entry in self.X_test]
        X_test = np.array(list(zip(tau_test, [0] * len(tau_test))))
        test_labels = self.kmeans.predict(X_test)

        self.X_test = np.delete(self.X_test, (-1), axis=1)
        self.X_test = np.c_[self.X_test, test_labels]
        # colors = ['r.','g.','b.','c.','k.','y.', 'm.'] * 10
        # for i in range(len(X_test)):
        #     plt.plot(X_test[i][0], X_test[i][1], colors[train_labels[i]], markersize=10)
        # plt.show()

    def save_kmeans(self, model_name):
        # pickle.dump(self.kmeans, open(model_name, 'wb'))
        joblib.dump(self.kmeans, model_name)


    def plot_confusion_matrix(self, model):
        test_predictions = model.predict(self.X_test)
        test_groundtruth = self.y_test
        conf_matrix = confusion_matrix(test_groundtruth, test_predictions)
        conf_matrix_norm = confusion_matrix(test_groundtruth, test_predictions, normalize='true')
        key_name = [key for key in KEYS]

        confmatrix_df = pd.DataFrame(conf_matrix, index=key_name, columns=key_name)
        confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=key_name, columns=key_name)

        # Plot confusion matrix.
        plt.figure(figsize=(16,6))
        sn.set(font_scale=0.8)
        plt.subplot(1,2,1)
        plt.title('Confusion Matrix')
        sn.heatmap(confmatrix_df, annot=True, annot_kws={'size':8})
        plt.subplot(1,2,2)
        sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={'size':8})

class TDOA:

    def __init__(self, filepath, MAX_SHIFT=100, delta_d = 0.30, cc_algorithm='cc', test=False):
        self.MAX_SHIFT = MAX_SHIFT
        self.cc_algorithm = cc_algorithm
        if not test:
            self.tau = self._get_tdoa(filepath)
        else:
            self.tau=self._get_tdoa_predict(filepath)

    def _get_tdoa_predict(self, filepath):
        y1, sr1 = librosa.load(filepath, sr=None)

        if 'sm7b' in filepath:
            f2 = 'audio/Interface/g0_nt1a.wav'
        else:
            f2 = 'audio/Interface/g0_sm7b.wav'
        y2, sr2 = librosa.load(f2, sr=None)
        cc, lag = self._xcorr(y1, y2)
        return np.argmax(cc) - self.MAX_SHIFT

    def _get_tdoa(self, filepath):
        y1, self.sr = librosa.load(filepath, sr=None)
        path, f = os.path.split(filepath)
        _, mic = os.path.split(path)
        mic = mic.strip()
        if mic == 'Rode NT1A':
            s = os.path.join('code', 'src', 'data', 'audio', os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1], 'Shure SM7B', f)
        else:
            s = os.path.join('code', 'src', 'data', 'audio', os.path.split(os.path.split(os.path.split(filepath)[0])[0])[1], 'Rode NT1A', f)
        y2, _ = librosa.load(s, sr=None)
        if self.sr != _ :
            raise ValueError('The sample rate of both files are different.')
        if self.cc_algorithm == 'cc':
            cc, lag = self._xcorr(y1, y2)
        if self.cc_algorithm == 'gcc':
            cc, lag = self._gccphat(y1, y2)
        return (np.argmax(cc) - self.MAX_SHIFT) / self.sr # divide by sr so able to compare at different srs [s]
        # return lag / self.sr

    # def _get_angle(self, filepath):
    #     V_LIGHT = 343 # approximate speed of light [m/s]
    #     tau_s = self.tau / self.sr

    #     AB_star = tau_s * V_LIGHT
    #     x_b = delta_d / 2
    #     x_more_than = (-1 * (AB_star**2 * (AB_star**2 - 4 * x_b**2)) / (4 * (4 * x_b**2 - AB_star**2)) )**0.5
    #     x_plot = np.arange(x_more_than+0.001, 0.10, 0.001)
    #     y_plot = np.array([])

    #     for x in x_plot:
    #         y = ( AB_star**2 / 4 - x_b**2 + x**2 * ((4 * x_b**2) / AB_star**2 - 1) )**0.5
    #         y_plot = np.append(y_plot, y)

    #     x_vals = np.where(x_plot > 0.02)[0]
    #     ar = np.where(x_plot>0.02)
    #     y_vals = y_plot[ar]
    #     dx = np.diff(x_vals)
    #     dy = np.diff(y_vals)
    #     # slope = np.mean(np.gradient(x_vals, y_vals))
    #     slope = np.nanmean(dy/dx) * 1000
    #     # if AB_star >= 0:
    #     #     pass
    #     # else:
    #     #     slope = -slope
    #     alpha_star = np.arctan(slope)
    #     alpha = np.degrees(np.pi/2 - alpha_star)

    #     if tau > 0:
    #         alpha = -alpha
    #     if tau == 0:
    #         alpha = 0
    #     if slope == np.nan:
    #         alpha=0
    #     if alpha == np.nan:
    #         alpha=0
    #     if np.absolute(tau*96000) >= 50:
    #         alpha=0
    #     print(alpha)
    #     return alpha

    def _xcorr(self, x, y):
        cc = np.correlate(x, y, 'full')
        zero_lag = np.size(cc) // 2
        cc = cc[zero_lag - self.MAX_SHIFT : zero_lag + self.MAX_SHIFT + 1]
        cc = ((cc - cc.min()) / (cc.max() - cc.min()))
        lag = np.argmax(cc) - zero_lag
        return cc, lag

    def _gccphat(self, x, y, interp=1):
        """https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py"""
        # to avoid time domain aliasing, use FFT size 'n' as large as shapes of x and y
        n = x.shape[0] + y.shape[0]
        x = np.fft.rfft(x, n=n)
        y = np.fft.rfft(y, n=n)
        R = x * np.conj(y)

        cc = np.fft.irfft(R / np.abs(R), n=(interp*n))
        max_shift = int(interp * n / 2)
        if self.MAX_SHIFT:
            max_shift=np.minimum(int(interp*self.sr*self.MAX_SHIFT), self.MAX_SHIFT)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
        # find max cross correlation indexo
        shift_plot = np.arange(-max_shift, max_shift+1)
        shift = np.argmax(cc) - max_shift
        return cc, shift

class CrossPredict:

    def __init__(self, prob_lst1, prob_lst2, y_true):
        global KEYS
        self.prob_lst1 = prob_lst1
        self.prob_lst2 = prob_lst2
        self.y_true = y_true

        if len(self.prob_lst1) != len(self.prob_lst2):
            raise ValueError('Lists are not equal length.')

    def cross_prediction(self):
        self.y_pred = []
        for i in range(len(self.prob_lst1)):
            d = {key: self.prob_lst1[i].get(key, 0) + self.prob_lst2[i].get(key, 0) \
                    for key in set(self.prob_lst1[i])}
            d_asc = sorted(d, key=d.get, reverse=True)
            self.y_pred.append(d_asc[0])

    def show_accuracy(self, confmatrix=True):
        self.cross_prediction()
        accuracy = accuracy_score(self.y_true, self.y_pred)
        print(f'Accuracy: {accuracy:.2f} when Cross Predicting.')

        self.ans = ''
        print('Keys Incorrectly Identified:')
        for i, key in enumerate(self.y_true):
            if key != self.y_pred[i]:
                # print(f'Predicted: {self.y_pred[i]} - Actual: {key}')
                # input()
                pass
            self.ans += self.y_pred[i]

        if confmatrix:
            self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        conf_matrix_norm = confusion_matrix(self.y_true, self.y_pred, normalize='true')
        key_name = [key for key in KEYS]

        confmatrix_df = pd.DataFrame(conf_matrix, index=key_name, columns=key_name)
        confmatrix_df_norm = pd.DataFrame(conf_matrix_norm, index=key_name, columns=key_name)

        # Plot confusion matrix.
        plt.figure(figsize=(16,6))
        sn.set(font_scale=0.8)
        plt.subplot(1,2,1)
        plt.title('Confusion Matrix')
        sn.heatmap(confmatrix_df, annot=True, annot_kws={'size':8})
        plt.subplot(1,2,2)
        sn.heatmap(confmatrix_df_norm, annot=True, annot_kws={'size':8})

def main():
    file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface', 'Shure SM7B')
    filenames = subset_data(file_directory, n_samples_per_class=20)

    print('Shure')
    keys_sm7b = KeyClusters(file_directory, filenames)
    keys_sm7b.ensemble_fit()
    d1 = keys_sm7b.to_probability_dict()

    # Save clustering model.
    model_name = os.path.join('code', 'finalised_kmeans_sm7b.pkl')
    # keys_sm7b.save_kmeans(model_name)
    keys_sm7b.save_kmeans(model_name)


    # Save classification model.
    model_name = os.path.join('code', 'finalised_model_sm7b.pkl')
    # pickle.dump(keys_sm7b.eclf, open(model_name, 'wb'))
    joblib.dump(keys_sm7b.eclf, model_name)

    print('Rode')
    file_directory = os.path.join('code', 'src', 'data', 'audio', 'Interface', 'Rode NT1A')
    keys_nt1a = KeyClusters(file_directory, filenames)
    keys_nt1a.ensemble_fit()
    d2 = keys_nt1a.to_probability_dict()

    # Save clustering model.
    model_name = os.path.join('code', 'finalised_kmeans_nt1a.pkl')
    keys_nt1a.save_kmeans(model_name)

    # Save classification model.
    model_name = os.path.join('code', 'finalised_model_nt1a.pkl')
    # pickle.dump(keys_nt1a.eclf, open(model_name, 'wb'))
    joblib.dump(keys_nt1a.eclf, model_name)

    # Cross compare both models to remove inaccuracies.
    cross_predict = CrossPredict(d1, d2, keys_nt1a.y_test)
    cross_predict.show_accuracy(confmatrix=False)
    plt.show()

if __name__ == "__main__":
    main()

