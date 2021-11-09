import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import librosa
import os
import joblib
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sklearn.exceptions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    silhouette_score
    )
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    GridSearchCV,
    train_test_split
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from slice_audio import DotDict
from tdoa import TDOA
import utils

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class LoadData:

    def __init__(self, *directories):
        self.directory = []
        for file_directory in directories:
            self.directory.append(file_directory)

    def load_data(self, sr=None, filters=None, n_samples_per_class=None,
            mono=False):
        self.tt_split=False
        self.filepaths = []
        for file_directory in self.directory:
            if n_samples_per_class == None:
                self.filepaths.append(utils._get_filepaths(file_directory,
                    filters=filters))
            else:
                self.filepaths.append(utils.subset_data(file_directory,
                    n_samples_per_class, filters=filters))
        self.filepaths = utils.flatten_list(self.filepaths)

        self.X, self.Y = [], []
        counter = 0
        for filepath in self.filepaths:
            key_id = utils._get_key_id(filepath)
            data = self._get_mfccs(filepath, sr=sr)
            # data = self._get_fft_coefficients(filepath)
            if not mono:
                tau = self._get_tau(filepath, sr=sr)
                data = np.append(data, tau)
            self.X.append(data)
            self.Y.append(key_id)
            counter += 1
        print(f'{counter}/{len(self.filepaths)} files processed.')

    def tt_split_data(self):
        self.tt_split=True
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.Y, test_size=0.2, random_state=10)

    def label_encode_data(self, export_to_file=None, ext='.bin'):
        self.le = LabelEncoder()
        self.le.fit(self.Y)
        self.Y = self.le.transform(self.Y)

        if export_to_file != None:
            joblib.dump(self.le.classes_, export_to_file + ext)

    def scale_data(self, export_to_file=None, ext='.bin'):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        if export_to_file != None:
            joblib.dump(self.scaler.scale_, export_to_file + '_scale' + ext)
            joblib.dump(self.scaler.mean_, export_to_file + '_mean' + ext)
            joblib.dump(self.scaler.var_, export_to_file + '_var' + ext)
            print(f'StandardScaler written to {export_to_file}(_scale, _mean, _var){ext}')

    def _get_mfccs(self, filepath, sr=None, duration=.2):
        y, self.sr = librosa.load(filepath, sr=None, duration=duration)
        y = librosa.util.normalize(y)

        n_fft = int(self.sr*0.01)
        y_mfcc = librosa.feature.mfcc(y, sr=self.sr, n_mels=16, fmin=400,
                fmax=12000, n_mfcc=13, n_fft=n_fft, hop_length=n_fft//4)
        y_d_mfcc = librosa.feature.delta(y_mfcc, mode='nearest')
        y_d2_mfcc = librosa.feature.delta(y_mfcc, order=2, mode='nearest')
        return y_mfcc.mean(1)

    def _get_fft_coefficients(self, filepath, sr=None, duration=.2):
        y, self.sr = librosa.load(filepath, sr=None, duration=duration)
        n_fft = int(0.08*self.sr)
        y_fft = librosa.stft(y, n_fft=n_fft, hop_length=n_fft//4)
        return np.abs(y_fft.mean(1))

    def _get_tau(self, filepath, max_shift=50, sr=None, cc_algo='gccphat'):
        ref_filepath = utils._get_ref_filepath(filepath)

        tdoa = TDOA(filepath, ref_filepath, sr=sr, max_shift=max_shift,
                cc_algo=cc_algo)
        return tdoa.get_tdoa()

    def cluster_tau(self, n_clusters, export_to_file=None, plot=True):
        if self.tt_split:
            # Get tau from train data.
            tau_train = [i[-1] for i in self.X_train]
            data_kmeans_train = np.array(list(zip(tau_train, [0] * len(tau_train))))
            self.kmeans = KMeans(n_clusters=n_clusters,
                    random_state=10).fit(data_kmeans_train)
            labels_train = self.kmeans.labels_
            self.X_train = np.c_[self.X_train, labels_train]

            # Get tau from test data.
            tau_test = [i[-1] for i in self.X_test]
            data_kmeans_test = np.array(list(zip(tau_test, [0] * len(tau_test))))
            labels_test = self.kmeans.predict(data_kmeans_test)
            self.X_test = np.c_[self.X_test, labels_test]

            joblib.dump(self.kmeans, export_to_file)

        else:
            tau = [i[-1] for i in self.X]
            data_kmeans = np.array(list(zip(tau, [0] * len(tau))))
            self.kmeans = joblib.load(export_to_file)
            labels = self.kmeans.predict(data_kmeans)
            self.X = np.c_[self.X, labels]
            print(self.X.shape)


        # Plot clusters.
        if plot:
            centroids = self.kmeans.cluster_centers_
            colors = ['r.','g.','b.','c.','k.','y.', 'm.'] * 10
            plt.figure(figsize=(10,10))
            for i in range(len(tau_train)):
                plt.plot(data_kmeans_train[i][0], data_kmeans_train[i][1],
                        colors[labels_train[i]], markersize=10)
                plt.annotate(self.y_train[i], (data_kmeans_train[i][0],
                    data_kmeans_train[i][1]+np.random.uniform(0, 0.05)))
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150,
                    linewidths=5)
            plt.show()

    def _silhouette_value(self, data, cluster_labels, n_clusters):
        silhouette_avg = silhouette_score(data, cluster_labels)
        return silhouette_avg

    def elbow_method_clusters(self, n_clusters_max):
        """
        Inspired from [1]:

            [1] https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
        """
        wcss, silhouette = [], []
        tau = [i[-1] for i in self.X_train]
        data_kmeans_train = np.array(list(zip(tau, [0] * len(tau))))
        for i in range(2, n_clusters_max+1):
            kmeans = KMeans(n_clusters=i, max_iter=300, random_state=10)
            kmeans.fit(data_kmeans_train)
            cluster_labels = kmeans.labels_
            sil = self._silhouette_value(data_kmeans_train, cluster_labels, i)
            wcss.append(kmeans.inertia_)
            silhouette.append(sil)

        # Normalize 'wcss' and 'silhouette' for visualisation purposes.
        wcss = [(float(i) - min(wcss)) / (max(wcss) - min(wcss)) for i in wcss]
        data = list(zip([i for i in range(2, n_clusters_max)], wcss, silhouette))
        df = pd.DataFrame(data, columns=['Cluster', 'WCSS', 'Silhouette'])

        # Plot optimum cluster analytics.
        fig, ax1 = plt.subplots(figsize=(12, 6))
        color='cadetblue'
        ax1.set_xlabel('Number of Clusters', fontsize=11)
        ax1.set_ylabel('Normalised Silhouette', fontsize=11, color=color)
        ax1 = sn.barplot(x='Cluster', y='Silhouette', data=df,
                alpha=.5, color=color)
        ax1.tick_params(axis='y')
        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalised WCSS %', fontsize=11, color=color)
        ax2 = sn.pointplot(x='Cluster', y='WCSS', data=df, marker='o',
                sort=False, color=color)
        ax2.tick_params(axis='y', color=color)

    def clusters_by_class(self):
        self.perc_df = []
        data = list(zip(self.kmeans.labels_, self.y_train))
        df = pd.DataFrame(data, columns=['Clusters', 'Labels'])
        for cluster in set(self.kmeans.labels_):
            df_cluster = df[df['Clusters'] == cluster]
            data = []
            for label in df_cluster['Labels'].unique():
                count = df_cluster[df_cluster['Labels'] == label].shape[0]
                total = df_cluster['Clusters'].shape[0]
                label = utils._list_to_str(self.le.inverse_transform([label]))
                data.append((label, count/total))
            perc_df = pd.DataFrame(data, columns=['Label', 'Percentage']).sort_values(
                    by='Percentage', axis=0, ascending=False)
            self.perc_df.append(perc_df)
            print(f'Cluster {cluster+1}')
            print(perc_df)
            # input()

    def to_probability_dict(self, clf, tt_split=True):
        d = []
        if tt_split:
            clf.fit(self.X_train, self.y_train)
            X = self.X_test
        else:
            X = self.X
        keys = self.le.inverse_transform(clf.classes_)
        for i in range(len(X)):
            vals = clf.predict_proba(X)[i]
            d.append(dict(zip(keys, vals)))
        return d

class Model(LoadData):

    def __init__(self, *directories):
        super().__init__(*directories)

    def metrics_report(self, *clfs, cv=5):
        scoring = {
                   'Accuracy': 'accuracy',
                   'Precision': 'precision_macro',
                   'Recall': 'recall_macro',
                   'F1': 'f1_macro'
                  }

        columns = utils.flatten_list(['Classifier', [k for k, _ in scoring.items()]])

        clf_reports = []
        for clf in clfs:
            clf_reports.append(self._get_clf_report(clf, scoring, cv=cv))

        report_df = pd.DataFrame(clf_reports, columns=columns).sort_values(
                by=columns[1], axis=0, ascending=False)

        print(f'{cv} Cross-Validated Scores:')
        print(report_df)
        return report_df

    def classify_by_clusters(self, n_classes_per_cluster):
        data_train = list(zip(self.X_train, self.y_train))
        data_test = list(zip(self.X_test, self.y_test))
        labels = self.kmeans.labels_

        scores = []
        for cluster in set(labels):
            # Get first n_classes_per_cluster from each cluster group.
            df = self.perc_df[cluster]
            cluster_keys = list(df['Label'][:n_classes_per_cluster])

            # Collect training data only belonging to cluster_keys.
            cluster_train_data = []
            cluster_test_data = []
            for key in self.le.transform(cluster_keys):
                for entry in data_train:
                    if entry[-1] == key:
                        cluster_train_data.append(entry)
            for i, entry in enumerate(data_test):
                if self.labels_test[i] == cluster:
                    cluster_test_data.append(entry)

            # Fit classifier to keys in each cluster.
            X_train, y_train = list(zip(*cluster_train_data))
            X_test, y_test = list(zip(*cluster_test_data))
            test_clf = RandomForestClassifier(n_estimators=100)
            test_clf.fit(X_train, y_train)

            # Only use test data which has respective cluster.
            score = test_clf.score(X_test, y_test)
            scores.append((cluster, score*100))
        report = pd.DataFrame(scores, columns=['Cluster', 'Accuracy'])
        print(report)

    def confusion_matrix(self, clf):
        clf.fit(list(self.X_train), self.y_train)
        conf_matrix = confusion_matrix(self.y_test, clf.predict(self.X_test))
        key_id_lst = sorted(set(self.Y))
        conf_matrix_df = pd.DataFrame(conf_matrix, index=key_id_lst, columns=key_id_lst)

        # Plot confusion matrix.
        plt.figure(figsize=(10,10))
        sn.set(font_scale=0.75)
        sn.heatmap(conf_matrix_df, annot=True, annot_kws={'size': 8}, cbar=False)

    def _clf_name(self, clf):
        clf_name = type(clf).__name__
        if clf_name == 'SVC' and clf.kernel == 'rbf':
            clf_name += ' RBF kernel'
        return clf_name


class CreateModel(Model):

    def __init__(self, *directories):
        super().__init__(*directories)

    def export_clf(self, clf, filename):
        clf.fit(self.X_train, self.y_train)
        joblib.dump(clf, filename)

    def gridsearch_tune_parameters(self, model, tuned_parameters_dict, scores):
        """https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html"""
        for score in scores:
            print(f'Tuning hyper-parameters for {score}')
            print()

            clf = GridSearchCV(model, tuned_parameters_dict, scoring=f'{score}_macro')
            clf.fit(self.X_train, self.y_train)

            print('Best parameters set found on development set:')
            print()
            print(clf.best_params_)
            print()
            print('Grid scores on development set:')
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print(f'{mean:.2f} (+/- {std*2:.2f}) for {params}')
            print()

            print('Detailed classification report:')
            print()
            print('The model is trained on the full development set.')
            print('The scores are computed on the full evaluation set.')
            y_true, y_pred = self.y_test, clf.predict(self.X_test)
            print(classification_report(y_true, y_pred))
            print()


    def _get_clf_report(self, clf, scoring, cv):
        clf_name = self._clf_name(clf)

        score = DotDict(cross_validate(
            clf, self.X, self.Y, scoring=scoring, cv=cv))

        return ((clf_name, score.test_Accuracy.mean(), score.test_Precision.mean(),
            score.test_Recall.mean(), score.test_F1.mean()))

    def ensemble_clf(self, *clfs, weights=None):
        estimators = [(self._clf_name(clf), clf) for clf in clfs]
        self.eclf = VotingClassifier(estimators=estimators, voting='soft')


class PredictFromModel(Model):

    def __init__(self, *directories):
        super().__init__(*directories)

    def import_clf(self, filename):
        self.clf = joblib.load(filename)

    def import_standard_scaler(self, filename, ext='.bin'):
        self.scaler = StandardScaler()
        self.scaler.n_features_in_ = len(self.X[0])
        self.scaler.scale_ = joblib.load(filename + '_scale' + ext)
        self.scaler.mean_ = joblib.load(filename + '_mean' + ext)
        self.scaler.var_ = joblib.load(filename + '_var' + ext)
        self.X = self.scaler.transform(self.X)

    def import_label_encoder(self, filename, ext='.bin'):
        self.le = LabelEncoder()
        self.le.classes_ = joblib.load(filename + ext)
        self.Y = self.le.transform(self.Y)

    def _get_clf_report(self):
        predictions = self.clf.predict(self.X)
        accuracy = accuracy_score(self.Y, predictions)
        precision = precision_score(self.Y, predictions, average='macro')
        recall = recall_score(self.Y, predictions, average='macro')
        f1 = f1_score(self.Y, predictions, average='macro')
        clf_name = self._clf_name(self.clf)
        df = pd.DataFrame([clf_name], columns=['Classifier'])
        df['Accuracy'], df['Precision'], df['Recall'], df['F1'] = [accuracy], [precision], [recall], [f1]
        return df

class CrossPredict(Model):

    def __init__(self, d1, d2, y_true):
        self.d1, self.d2, self.y_true = d1, d2, y_true

        if len(self.d1) != len(self.d2):
            raise ValueError('Probabaility dictionaries are not equal length.')

    def _cross_predict(self, weights):
        self.y_pred = []
        for i in range(len(self.d1)):
            d = {key: self.d1[i].get(key, 0)*weights[0] + self.d2[i].get(key,
                0)*weights[1]
                    for key in set(self.d1[i])}
            d = sorted(d, key=d.get, reverse=True)
            self.y_pred.append(d[0])

    def get_accuracy(self, weights=[0.5, 0.5], show_incorrect_predictions=True,
            confmatrix=False):
        self._cross_predict(weights)
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred, average='macro')
        recall = recall_score(self.y_true, self.y_pred, average='macro')
        f1 = f1_score(self.y_true, self.y_pred, average='macro')
        df = pd.DataFrame()
        df['Accuracy'], df['Precision'], df['Recall'], df['F1'] = [accuracy], [precision], [recall], [f1]

        # print(f'Accuracy: {accuracy:.2f} (Cross Prediction)')
        if show_incorrect_predictions:
            print('Keys incorrectly identified:')
            for i, key in enumerate(self.y_true):
                if key != self.y_pred[i]:
                    print(f'Predicted: {self.y_pred[i]} -> Actual: {key}')

        if confmatrix:
            self._confusion_matrix()

        return df

    def _confusion_matrix(self):
        key_id_lst = set(self.y_true)
        key_id_lst.update(self.y_pred)
        key_id_lst = sorted(key_id_lst)
        conf_matrix = confusion_matrix(self.y_true, self.y_pred, normalize='true')

        conf_matrix_df = pd.DataFrame(conf_matrix, index=key_id_lst, columns=key_id_lst)
        plt.figure()
        plt.tight_layout()
        sn.set(font_scale=0.8)
        plt.title('Confusion Matrix User 1')
        sn.heatmap(conf_matrix_df, annot=True, annot_kws={'size':8})
        plt.savefig('dissertation_article/res/img/conf_matrix_user01', dpi=1200)

def gridsearch():
    svc = SVC()
    svc_tuned_parameters = [
        {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
        },
        {
        'kernel': ['linear'],
        'C': [1, 10, 100, 1000]
        }]

    rfc = RandomForestClassifier()
    rfc_tuned_parameters = [
        {
        'n_estimators': [70, 80, 90, 100, 110],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10]
        }]

    mlp = MLPClassifier()
    mlp_tuned_parameters = [
        {
        'learning_rate': ['constant'],
        'hidden_layer_sizes': [(500, 400, 300, 200, 100),
                               (400, 400, 400, 400, 400),
                               (300, 300, 300, 300, 300),
                               (200, 200, 200, 200, 200)],

        'activation': ['identity'],
        'alpha': [0.0001, 0.001, 0.005],
        'early_stopping': [True, False]
        }]

    sm7b.gridsearch_tune_parameters(svc,
            svc_tuned_parameters, ['precision', 'recall'])
    sm7b.gridsearch_tune_parameters(rfc,
            rfc_tuned_parameters, ['precision', 'recall'])
    sm7b.gridsearch_tune_parameters(mlp,
            mlp_tuned_parameters, ['precision', 'recall'])

def main_create():
    clf_directory = os.path.join('src', 'data', 'models')
    df_dir = os.path.join('src', 'data', 'dataframes')
    parent_directory = os.path.join('src', 'data', 'audio')

    file_directory_sm7b = os.path.join(parent_directory, 'Interface', 'CH03')
    file_directory_nt1a = os.path.join(parent_directory, 'Interface', 'CH02')

    # Load and scale data.
    sm7b = CreateModel(file_directory_sm7b)
    sm7b.load_data(sr=24000)
    sm7b.label_encode_data(export_to_file='encoder_sm7b')
    sm7b.tt_split_data()
    sm7b.scale_data(export_to_file='scaler_sm7b')

    nt1a = CreateModel(file_directory_nt1a)
    nt1a.load_data(sr=24000)
    nt1a.label_encode_data(export_to_file='encoder_nt1a')
    nt1a.tt_split_data()
    nt1a.scale_data(export_to_file='scaler_nt1a')
    # nt1a.cluster_tau(n_clusters=7,
    #         export_to_file='code/models/kmeans_nt1a.sav', plot=False)
    # nt1a.clusters_by_class()
    # nt1a.classify_by_clusters(n_classes_per_cluster=5)
    # sm7b.elbow_method_clusters(n_clusters_max=20)
    # plt.show()

    # Create models (best parameters found using GridSearchCV).
    nb = GaussianNB()
    svc = SVC(kernel='rbf', C=100, gamma=0.0001, probability=True)
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10,
            min_samples_split=2, min_samples_leaf=1)
    sm7b.ensemble_clf(nb, svc, rfc, weights=[0.3, 0.2, 0.5])
    nt1a.ensemble_clf(nb, svc, rfc, weights=[0.3, 0.2, 0.5])

    # Report metrics.
    print()
    print('Shure SM7B')
    sm7b_metrics_df = sm7b.metrics_report(nb, svc, rfc, sm7b.eclf)
    sm7b.export_clf(rfc, filename=os.path.join(clf_directory, 'RFC_sm7b.sav'))

    # sm7b.confusion_matrix(rfc)
    print()
    print('Sonarworks XREF20')
    nt1a_metrics_df = nt1a.metrics_report(nb, svc, rfc, nt1a.eclf)

    d_sm7b = sm7b.to_probability_dict(sm7b.eclf)
    d_nt1a = nt1a.to_probability_dict(nt1a.eclf)
    nt1a.export_clf(rfc, filename=os.path.join(clf_directory, 'RFC_nt1a.sav'))

    print()
    clfs = [nb, svc, rfc, 'eclf']
    df = []
    for clf in clfs:
        if clf == 'eclf':
            d_sm7b = sm7b.to_probability_dict(sm7b.eclf)
            d_nt1a = nt1a.to_probability_dict(nt1a.eclf)
        else:
            d_sm7b = sm7b.to_probability_dict(clf)
            d_nt1a = nt1a.to_probability_dict(clf)

        y_true = sm7b.le.inverse_transform(sm7b.y_test)
        cpred = CrossPredict(d_sm7b, d_nt1a, y_true)
        df_clf = cpred.get_accuracy(show_incorrect_predictions=False)
        df_clf['Classifier'] = ['VotingClassifier' if clf == 'eclf' else cpred._clf_name(clf)]
        df.append(df_clf)
    cpred_metrics_df = pd.concat(df)

    print()
    print('Cross-Prediction Metrics')
    print(cpred_metrics_df)

    # Export eclf models.
    sm7b.export_clf(sm7b.eclf, filename=os.path.join(clf_directory, 'ECLF_sm7b.sav'))
    nt1a.export_clf(nt1a.eclf, filename=os.path.join(clf_directory, 'ECLF_nt1a.sav'))

def main_predict():
    user, classifier, training_from, typing_style = 'USER01', 'RFC', 'TT_48', 'SAMPLE_TEXT'
    df_dir = os.path.join('src', 'data', 'dataframes', user, classifier, training_from)

    parent_directory = os.path.join('src', 'data', 'audio', user, typing_style)
    file_directory = os.path.join(parent_directory, 'STEREO', 'CH03')

    sm7b = PredictFromModel(file_directory)
    sm7b.load_data(filters=['1','2','3','4','5','6','7','8','9','0',
        'Key.esc', 'Key.backspace', 'Key.enter', ','], sr=48000)

    sm7b.import_clf(f'code/src/data/models/{classifier}_sm7b.sav')
    print(sm7b._clf_name(sm7b.clf))

    sm7b.import_label_encoder('code/encoder_sm7b')
    sm7b.import_standard_scaler('code/scaler_sm7b')
    df_sm7b = sm7b._get_clf_report()
    filepath_sm7b = os.path.join(df_dir, f'{typing_style}_SM7B.pkl')
    df_sm7b.to_pickle(filepath_sm7b)
    print(df_sm7b)

    file_directory = os.path.join(parent_directory, 'STEREO', 'CH02')
    nt1a = PredictFromModel(file_directory)
    nt1a.load_data(filters=['1','2','3','4','5','6','7','8','9','0',
        'Key.esc', 'Key.backspace', 'Key.enter', ','], sr=48000)

    nt1a.import_clf(f'code/src/data/models/{classifier}_nt1a.sav')
    nt1a.import_label_encoder('code/encoder_nt1a')
    nt1a.import_standard_scaler('code/scaler_nt1a')
    nt1a._get_clf_report()
    df_nt1a = nt1a._get_clf_report()
    filepath_nt1a = os.path.join(df_dir, f'{typing_style}_XREF.pkl')
    df_nt1a.to_pickle(filepath_nt1a)
    print(df_nt1a)

    d_sm7b = sm7b.to_probability_dict(sm7b.clf, tt_split=False)
    d_nt1a = nt1a.to_probability_dict(nt1a.clf, tt_split=False)
    y_test = sm7b.le.inverse_transform(sm7b.Y)
    df_cpred = CrossPredict(d_sm7b, d_nt1a,
            y_test).get_accuracy(show_incorrect_predictions=True, confmatrix=True)
    filepath_cpred = os.path.join(df_dir, f'{typing_style}_CPRED.pkl')
    df_cpred['Classifier'] = [sm7b._clf_name(sm7b.clf)] * df_cpred.shape[0]
    df_cpred.to_pickle(filepath_cpred)
    print(df_cpred)
    plt.show()

if __name__ == '__main__':
    # gridsearch()
    main_create()
    # main_predict()
