import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from enum import Enum
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import neighbors
from surprise import Reader, Dataset, KNNWithMeans, SVD, NMF, accuracy
from surprise.model_selection import train_test_split
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import calculations_no_item_filtering as calcs


class Imputer:
    # this code is copied from:
    # https://github.com/bwanglzu/Imputer.py
    """Imputer class."""

    def _fit(self, X, column, k=10, is_categorical=False):
        """Fit a knn classifier for missing column.
        - Args:
                X(numpy.ndarray): input data
                column(int): column id to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                clf: trained k nearest neighbour classifier
        """
        clf = None
        if not is_categorical:
            clf = neighbors.KNeighborsRegressor(n_neighbors=k)
        else:
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        # use column not null to train the kNN classifier
        missing_idxes = np.where(pd.isnull(X[:, column]))[0]
        if len(missing_idxes) == 0:
            return None
        X_copy = np.delete(X, missing_idxes, 0)
        X_train = np.delete(X_copy, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_train[:, col_id]))[0]
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_train[col_missing_idxes, col_id] = col_mean[col_id]
        y_train = X_copy[:, column]
        # fit classifier
        clf.fit(X_train, y_train)
        return clf

    def _transform(self, X, column, clf, is_categorical):
        """Impute missing values.
        - Args:
                X(numpy.ndarray): input numpy ndarray
                column(int): index of column to be imputed
                clf: pretrained classifier
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                X(pandas.dataframe): imputed dataframe
        """
        missing_idxes = np.where(np.isnan(X[:, column]))[0]
        X_test = X[missing_idxes, :]
        X_test = np.delete(X_test, column, 1)
        # if other columns still have missing values fill with mean
        col_mean = None
        if not is_categorical:
            col_mean = np.nanmean(X, 0)
        else:
            col_mean = np.nanmedian(X, 0)
        # fill missing values in each column with current col_mean
        for col_id in range(0, len(col_mean) - 1):
            col_missing_idxes = np.where(np.isnan(X_test[:, col_id]))[0]
            # if no missing values for current column
            if len(col_missing_idxes) == 0:
                continue
            else:
                X_test[col_missing_idxes, col_id] = col_mean[col_id]
        # predict missing values
        y_test = clf.predict(X_test)
        X[missing_idxes, column] = y_test
        return X

    def knn(self, X, column, k=10, is_categorical=False):
        """Impute missing value with knn.
        - Args:
                X(pandas.dataframe): dataframe
                column(str): column name to be imputed
                k(int): number of nearest neighbors, default 10
                is_categorical(boolean): is continuous or categorical feature
        - Returns:
                NOT WORKING: X_imputed(pandas.dataframe): imputed pandas dataframe (originally)

                Changed by Ognjen Francuski:
                X_imputed(nd.array): IMPUTED VALUES FOR PASSED COLUMN
        """
        X, column = self._check_X_y(X, column)
        clf = self._fit(X, column, k, is_categorical)
        if clf is None:
            return X[:, column]
        else:
            X_imputed = self._transform(X, column, clf, is_categorical)
            return X_imputed[:, column]

    def _check_X_y(self, X, column):
        """Check input, if pandas.dataframe, transform to numpy array.
        - Args:
                X(ndarray/pandas.dataframe): input instances
                column(str/int): column index or column name
        - Returns:
                X(ndarray): input instances
        """
        column_idx = None
        if isinstance(X, DataFrame):
            if isinstance(column, str):
                # get index of current column
                column_idx = X.columns.get_loc(column)
            else:
                column_idx = column
            X = X.as_matrix()
        else:
            column_idx = column
        return X, column_idx


class Encoder:

    def __init__(self):
        self.label_encoders = defaultdict()


    def encode_labels(self, df, y_column, encoding_labels=None) -> DataFrame:
        encoded_df = df.copy(deep=True)
        if y_column is not None:
            self.numeric_encoding(encoded_df, [y_column])
        if encoding_labels is not None:
            encoded_df = self.one_hot_encoding(encoded_df, encoding_labels)
        return encoded_df


    def numeric_encoding(self, df, columns):
        for col in columns:
            if col not in self.label_encoders.keys():
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].astype(str))
            df[col] = self.label_encoders[col].transform(df[col].astype(str))


    def one_hot_encoding(self, df: DataFrame, columns, drop_one_col=True):
        return pd.get_dummies(df, columns=columns, dummy_na=False, drop_first=drop_one_col)

class ScalingType(Enum):
    Z_SCORE = 0
    MIN_MAX = 1


class DataPreprocessing:

    def __init__(self, train_set, test_set, normalization: ScalingType = ScalingType.Z_SCORE):
        self.train_set = train_set
        self.test_set = test_set
        self.encoder = Encoder()
        self.imputer = Imputer()
        self.is_scaler_fitted = False
        if normalization == ScalingType.Z_SCORE:
            self.scaler = StandardScaler()
        elif normalization == ScalingType.MIN_MAX:
            self.scaler = MinMaxScaler()

    # MISSING VALUES ===================================================================================================

    @staticmethod
    def get_rows_with_missing_values(df: DataFrame) -> DataFrame:
        return df[df.isnull().any(axis=1)]


    @staticmethod
    def drop_missing_values(df: DataFrame) -> DataFrame:
        return df.dropna(axis=0)


    @staticmethod
    def set_outlier_vales_as_nan(df: DataFrame, outliers_labels_dict):
        if outliers_labels_dict is None:
            return
        for column in outliers_labels_dict:
            df[column] = df[column].apply(
                lambda x: x if outliers_labels_dict[column][0] < x < outliers_labels_dict[column][1] else np.nan)


    def impute(self, df: DataFrame, cat_columns, y_column) -> DataFrame:

        encoding_dict = defaultdict()
        decoding_dict = defaultdict()

        cat_list = copy.deepcopy(cat_columns) if cat_columns is not None else []
        if y_column is not None and df[y_column].dtype == 'object':
            cat_list.append(y_column)

        cont_list = []
        for c in df.columns.values:
            if c not in cat_list and c != y_column:
                cont_list.append(c)

        for c in cat_list:
            if df[c].dtype != 'object':
                continue
            encoded_dict = defaultdict()
            decoded_dict = defaultdict()
            counter = 0
            unique_values = df[c].unique()
            for u in unique_values:
                if not isinstance(u, str) and np.isnan(u):
                    continue
                encoded_dict[u] = counter
                decoded_dict[counter] = u
                counter += 1
            encoding_dict[c] = encoded_dict
            decoding_dict[c] = decoded_dict

        copy_df = df.copy(deep=True)

        # do manual label encoding to keep nan values
        copy_df = copy_df.replace(to_replace=encoding_dict)

        for c in cat_list:
            copy_df[c] = self.imputer.knn(copy_df, c, is_categorical=True)
        for c in cont_list:
            copy_df[c] = self.imputer.knn(copy_df, c, is_categorical=False)

        # replace dataframe with imputed numerical values to original values
        copy_df = copy_df.replace(to_replace=decoding_dict)

        return copy_df


    # ENCODING =========================================================================================================

    def encode_labels(self, df, y_column, encoding_labels=None) -> DataFrame:
        return self.encoder.encode_labels(df, y_column, encoding_labels)


    # NORMALIZATION / DENORMALIZATION ==================================================================================

    def normalize(self, x, y=None):
        if not self.is_scaler_fitted:
            self.scaler.fit(x, y)
            self.is_scaler_fitted = True
        return self.scaler.transform(x)


    def denormalize(self, x):
        return self.scaler.inverse_transform(x)


    # FEATURE SELECTION ================================================================================================

    @staticmethod
    def feature_selection(dataset: DataFrame, dropping_features=None) -> DataFrame:
        dropping_features = [] if dropping_features is None else dropping_features
        return dataset.drop(dropping_features, axis=1)


    # MAIN PART ========================================================================================================

    def preprocess_data(self, cat_ohe_labels=None, cat_enc_labels=None, outlier_labels=None, dropping_features=None, y_label=None):

        dataset = pd.read_csv(self.train_set, delimiter="\t", names=["uid", "tid", "count"])
        testset = pd.read_csv(self.test_set, delimiter="\t", names=["uid", "tid", "count"])
        # dataset_and_testset = pd.concat([dataset, testset]).reset_index(drop=True)

        # TODO 1: FEATURE SELECTION
        # dataset_and_testset = self.feature_selection(dataset_and_testset, dropping_features)

        # TODO 1.5: SPLIT DATASET AND TESTSET TO REMOVE OUTLIERS ONLY FROM DATASET
        # dataset_and_testset = np.split(dataset_and_testset, [len(dataset)])
        # dataset = dataset_and_testset[0]
        # testset = dataset_and_testset[1].reset_index(drop=True)

        # TODO 2: DEAL WITH MISSING VALUES ON DATASET AND TESTSET SEPARATELY
        # imputing = [l for l in cat_ohe_labels]
        # imputing.extend(cat_enc_labels)
        # dataset = self.impute(df=dataset, cat_columns=imputing, y_column=y_label)
        # testset = self.impute(df=testset, cat_columns=imputing, y_column=y_label)

        # TODO 2.1: CONCATENATE SETS AGAIN TO DO PROPER LABEL AND ONE HOT ENCODING
        # dataset_and_testset = pd.concat([dataset, testset]).reset_index(drop=True)

        # TODO 3: ENCODING
        # dataset_and_testset = self.encoder.one_hot_encoding(dataset_and_testset, cat_ohe_labels, False)
        # self.encoder.numeric_encoding(dataset_and_testset, cat_enc_labels)
        # self.encoder.numeric_encoding(dataset_and_testset, [y_label])

        # correlation matrix said that this column affects race much less than never married (around 0.5% correlation compared to 5%)
        # del dataset_and_testset['maritl_5. Separated']

        # TODO: 3.1: SPLIT DATASET AND TESTSET TO FORMER PARTS
        # dataset_and_testset = np.split(dataset_and_testset, [len(dataset)])
        # dataset = dataset_and_testset[0]
        # testset = dataset_and_testset[1].reset_index(drop=True)

        # TODO 3.2: CREATE x AND y FOR DATASET AND TESTSET
        x = dataset.loc[:, dataset.columns != y_label].values
        y = dataset[y_label].values
        x_test = testset.loc[:, testset.columns != y_label].values
        y_test = testset[y_label].values

        # TODO 4: NORMALIZATION
        # x_normalized = self.normalize(x)
        # x_test_normalized = self.normalize(x_test)

        # TODO 5: REMOVE OUTLIERS
        # lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=-1)
        # y_pred = lof.fit_predict(x_normalized)
        # y_pred = lof.fit_predict(x)
        # x_normalized = np.array([x_normalized[i] for i in range(len(y_pred)) if y_pred[i] == 1])
        # x_normalized = np.array([x[i] for i in range(len(y_pred)) if y_pred[i] == 1])
        # y = np.array([y[i] for i in range(len(y_pred)) if y_pred[i] == 1])

        return x, y, x_test, y_test # x_normalized, y, x_test_normalized, y_test

def main():
    train_path = 'data/10000.txt'
    test_path = 'data/year1_test_triplets_visible-test.txt'
    final_test_path = 'data/year1_test_triplets_hidden-test.txt'

    dataset = pd.read_csv(train_path, delimiter="\t", names=["uid", "tid", "count"])
    testset = pd.read_csv(test_path, delimiter="\t", names=["uid", "tid", "count"])
    final_testset = pd.read_csv(final_test_path, delimiter="\t", names=["uid", "tid", "count"])
    dataset_and_testset = pd.concat([dataset, testset]).reset_index(drop=True)
    dataset_and_testset2 = pd.concat([dataset_and_testset, final_testset]).reset_index(drop=True)

    # dp = DataPreprocessing(train_set=train_path, test_set=test_path, normalization=ScalingType.Z_SCORE)
    # x_normalized, y, x_test_normalized, y_test = dp.preprocess_data(cat_ohe_labels=[],
    #                                                                 cat_enc_labels=[],
    #                                                                 y_label="count",
    #                                                                 dropping_features=[])
    # data = ColumnarModelData.from_data_frame(path, val_indx, x, y, [‘userId’, ‘movieId’], 64)
    # uid tid count
    # print(x_normalized, y, x_test_normalized, y_test)

    reader = Reader()
    data = Dataset.load_from_df(dataset_and_testset[['uid', 'tid', 'count']], reader)
    # test = Dataset.load_from_df(final_testset[['uid', 'tid', 'count']], reader)

    trainset, testset = train_test_split(data, test_size=len(final_testset)/len(dataset_and_testset2))


    # param_grid = {#'n_factors': [110, 160],
    #               'n_epochs': [90, 110],
    #               'lr_all': [0.001, 0.008],
    #               # 'reg_all': [0.08, 0.15]
    # }
    # gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
    # # algo = SVD()
    # # print(model_selection.cross_validate(algo, data, measures=['RMSE']))
    # # # evaluate(algo, data, measures=['RMSE'])
    # gs.fit(data)
    # algo = gs.best_estimator['rmse']
    # print(gs.best_score['rmse'])
    # print(gs.best_params['rmse'])
    # # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # # # Matrix Factorization method
    # # Use the new parameters with the train data
    print("SVD", datetime.now())
    algo = SVD(n_factors=110, n_epochs=100, lr_all=0.005, reg_all=0.15)
    algo.fit(trainset)
    predictions = algo.test(testset)
    # accuracy.rmse(test_pred, verbose=True)

    # print("NMF")
    # param_grid = {
    #     'n_factors': [10, 20],
    #     'n_epochs': [40, 60],
    # }
    # gs = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=5)
    # gs.fit(data)
    # algo = gs.best_estimator['rmse']
    # print(gs.best_score['rmse'])
    # print(gs.best_params['rmse'])

    # print("NMF", datetime.now())
    # algo = NMF(n_factors=20, n_epochs=40)
    # # # evaluate(algo, data, measures=['RMSE'])
    # # print(model_selection.cross_validate(algo, data, measures=['RMSE']))
    # algo.fit(trainset)
    # predictions = algo.test(testset)
    # # print(accuracy.rmse(predictions), datetime.now())

    # item - item methods, user based ne zato sto previse memorije zauzima
    # item - item je onaj fazon mozda ce vam se svidjeti
    # algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
    # print("KNNWithMeans item_based", datetime.now())
    # # evaluate(algo, data, measures=['RMSE'])
    # print(model_selection.cross_validate(algo, data, measures=['RMSE']))
    # algo.fit(trainset)
    # predictions = algo.test(testset)

    # print(predictions[0])

    # print(accuracy.rmse(predictions), datetime.now())

    finalListeners = final_testset.groupby('uid', as_index=True).agg(lambda x: x.tolist()).to_dict()['tid']
    listenersRecs = {}
    for user in finalListeners:
        for p in find_prediction(user, predictions):
            if p.uid in listenersRecs:
                listenersRecs[p.uid].append(p.iid)
            else:
                listenersRecs[p.uid] = [p.iid]
    # print(final_testset.head())

    # print(listenersRecs)
    # print(finalListeners)

    print("Calculating MAP @", datetime.now())
    mapr = calcs.calcMeanAveragePrecision(listenersRecs, finalListeners)
    print("-----MAP-----")
    print(mapr)

    print("Calculating NDCG @", datetime.now())
    ndcg = calcs.calcNdcg(listenersRecs, finalListeners)
    print("-----NDCG ----")
    print(ndcg)

def find_prediction(id, predictions):
    ls = []
    for i in predictions:
        if str(i.uid) == str(id):
            ls.append(i)
    return ls

if __name__ == '__main__':
    main()
    print("main end ", datetime.now())
