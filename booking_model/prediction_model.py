# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer, ndcg_score
from sklearn.model_selection import train_test_split

# for modeling

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from catboost import CatBoostClassifier


def load_data():
    train_data = pd.read_csv('booking_model/data/train_users_2.csv.zip')
    test_data = pd.read_csv('booking_model/data/test_users.csv.zip')
    return train_data, test_data

class PredictionModel:

    def __init__(self):
        self.label_enc = LabelEncoder()
        self.ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def preprocessing(self, data):
        # data preprocessing
        print("##### data preprocessing #####")
        X = data.drop(columns=['country_destination'], axis=1).copy()
        y = data['country_destination'].copy()
        y = self.label_enc.fit_transform(y)
        print(X.shape, y.shape)
        print(self.label_enc.classes_)
        return X, y


    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test

    def get_transformed_data(self, X_train, X_test):

        # X_train, X_test, y_train, y_test = split_train_test(X, y)

        cat_attrs = ['gender', 'language', 'affiliate_channel', 'affiliate_provider']

        pre_process = ColumnTransformer([('drop_cols', 'drop', ['id', 'date_first_booking', 'date_account_created', 'signup_method', 'timestamp_first_active', 
                                                            'signup_app', 'first_device_type', 'first_browser', 'first_affiliate_tracked', 'signup_flow']),
                                    ('num_imputer', SimpleImputer(strategy='median'), ['age']),
                                    ('cat_imputer', SimpleImputer(strategy='most_frequent'), cat_attrs)], remainder='passthrough')

        X_train_transformed = pre_process.fit_transform(X_train)
        X_test_transformed = pre_process.transform(X_test)
        print(X_train_transformed.shape, X_test_transformed.shape)

        X_train_transformed = pd.DataFrame(X_train_transformed, columns=['age', 'gender', 'language', 'affiliate_channel', 'affiliate_provider'])
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=['age', 'gender', 'language', 'affiliate_channel', 'affiliate_provider'])
        print(X_train_transformed.shape, X_test_transformed.shape)

        return X_train_transformed, X_test_transformed, pre_process


    def dcg_score(self, y_true, y_score, k=5):
        """Discounted cumulative gain (DCG) at rank K.

        Parameters
        ----------
        y_true : array, shape = [n_samples]
            Ground truth (true relevance labels).
        y_score : array, shape = [n_samples, n_classes]
            Predicted scores.
        k : int
            Rank.

        Returns
        -------
        score : float
        """
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])

        gain = 2 ** y_true - 1

        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)


    def ndcg_score(self, ground_truth, predictions, k=5):
        """Normalized discounted cumulative gain (NDCG) at rank K.

        Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
        recommendation system based on the graded relevance of the recommended
        entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
        ranking of the entities.

        Parameters
        ----------
        ground_truth : array, shape = [n_samples]
            Ground truth (true labels represended as integers).
        predictions : array, shape = [n_samples, n_classes]
            Predicted probabilities.
        k : int
            Rank.

        Returns
        -------
        score : float

        Example
        -------
        >>> ground_truth = [1, 0, 2]
        >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        1.0
        >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        >>> score = ndcg_score(ground_truth, predictions, k=2)
        0.6666666666
        """
        lb = LabelBinarizer()
        lb.fit(range(len(predictions) + 1))
        T = lb.transform(ground_truth)

        scores = []

        # Iterate over each y_true and compute the DCG score
        for y_true, y_score in zip(T, predictions):
            actual = self.dcg_score(y_true, y_score, k)
            best = self.dcg_score(y_true, y_true, k)
            score = float(actual) / float(best)
            scores.append(score)

        return np.mean(scores)

    def grid_search(self, model, grid_param):
        
        
        print("Obtaining Best Model for {}".format(model.__class__.__name__))
        grid_search = GridSearchCV(model, grid_param, cv=self.kf, scoring=self.ndcg_scorer, return_train_score=True, n_jobs=-1)
        grid_search.fit(X_train_transformed, y_train)
        
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        
        cvres = grid_search.cv_results_
        print("Results for each run of {}...".format(model.__class__.__name__))
        for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):
            print(train_mean_score, test_mean_score, params)
            
        return grid_search.best_estimator_

        
    def performance_measures(self, model, store_results=True):
        train_ndcg = cross_val_score(model, X_train_transformed, y_train, scoring=self.ndcg_scorer, cv=self.kf, n_jobs=-1)
        test_ndcg = cross_val_score(model, X_test_transformed, y_test, scoring=self.ndcg_scorer, cv=self.kf, n_jobs=-1)
        print("Mean Train NDGC: {}\nMean Test NDGC: {}".format(train_ndcg.mean(), test_ndcg.mean()))

    def create_model(self, pre_process, X_train_transformed, y_train, X_train):

        catboost_grid_params = [{'iterations':[500, 1000, 1500], 'depth':[4, 6, 8, 10],}]

        catboost_clf = CatBoostClassifier(task_type="GPU", loss_function='MultiClass', bagging_temperature=0.3, 
                                        cat_features=[1, 2, 3, 4], random_state=42, verbose=0)

        grid_search_results = catboost_clf.grid_search(catboost_grid_params,
                    X_train_transformed,
                    y_train,
                    cv=5,
                    partition_random_seed=42,
                    calc_cv_statistics=True,
                    search_by_train_test_split=True,
                    refit=True,
                    shuffle=True,
                    stratified=None,
                    train_size=0.8,
                    verbose=0,
                    plot=False)

        print("##### Result for grid search for result ####")
        print(grid_search_results['params'])

        print(catboost_clf.is_fitted())

        print(self.performance_measures(catboost_clf, store_results=False))

        # fit models

        final_model = Pipeline([('pre_process', pre_process),
                            ('catboost_clf', catboost_clf)])
        final_model.fit(X_train, y_train)

        model_storage_path = "booking_model/artifacts"
        # store models
        import pickle
        with open(os.path.join(model_storage_path, 'model.pkl'), 'wb') as file:
            pickle.dump(final_model, file)
            file.close
        
        print("model storage completed")


    def predict_contries(self, test_data):
        import pickle
        model_storage_path = "booking_model/artifacts"
        with open(os.path.join(model_storage_path, 'model.pkl', 'rb')) as file:
            final_model = pickle.load(file)
        
        predictions = final_model.predict_proba(test_data)

        #Taking the 5 classes with highest probabilities
        id_test = list(test_data.id)
        ids = []
        countries = []
        for i in range(len(id_test)):
            idx = id_test[i]
            ids += [idx] * 5
            countries += self.label_enc.inverse_transform(np.argsort(predictions[i])[::-1])[:5].tolist()

        output = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])

        return output

