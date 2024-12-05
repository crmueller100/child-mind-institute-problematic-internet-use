# %load_ext cudf.pandas
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score # To calculate quadratic weighted Kappa
from sklearn.pipeline import Pipeline


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingClassifier


import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_dater():
    # print(f"Running loading dater()")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    data_dictionary = pd.read_csv('data_dictionary.csv')

    def process_file(filename, dirname):
        data = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
        data.drop('step', axis=1, inplace=True)
        return data.describe().values.reshape(-1), filename.split('=')[1]

    def load_time_series(dirname) -> pd.DataFrame:
        ids = [d for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
        
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
        stats, indexes = zip(*results)
        
        data = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
        data['id'] = indexes
        return data

    def save_time_series_as_df():
        train_ts = load_time_series('./series_train.parquet')
        test_ts = load_time_series('./series_test.parquet')
        train_ts.to_csv('train_ts.csv', index=False)
        test_ts.to_csv('test_ts.csv', index=False)

    # comment this after you run it once
    # save_time_series_as_df()

    # print("Reading from saved time series data")
    train_ts = pd.read_csv('train_ts.csv')
    test_ts = pd.read_csv('test_ts.csv')


    train_df = pd.merge(train_df, train_ts, how="left", on='id')
    test_df = pd.merge(test_df, test_ts, how="left", on='id')

    # Figure this was safe to do now. I moved it up from lower
    def combine_fe_time_into_one_column(df):
        # check if either column is null
        null_mask = df['Fitness_Endurance-Time_Mins'].isnull() | df['Fitness_Endurance-Time_Sec'].isnull()
        df['Fitness_Endurance-Time'] = df['Fitness_Endurance-Time_Mins'] + df['Fitness_Endurance-Time_Sec'] / 60

        # set result to null if either column is null
        df.loc[null_mask, 'Fitness_Endurance-Time'] = np.nan  

        df.drop(columns=['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec'], inplace=True)

    combine_fe_time_into_one_column(test_df)
    combine_fe_time_into_one_column(train_df)



    return train_df, test_df, train_ts, test_ts


def fill_in_nans_on_time_series_data(train_df, test_df, train_ts, test_ts):
    # print(f"running fill_in_nans_on_time_series_data()")
    time_series_cols = train_ts.columns.tolist()
    time_series_cols.remove('id')
    for col in time_series_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    return train_df, test_df, train_ds, test_ds

def remove_low_correlation_columns(train_df, test_df):
    # print(f"Running remove_low_correlation_columns()")

    # TODO: Experiment with different imputation strategies (or maybe none at all)
    df_corr = train_df.copy()
    df_corr['sii'] = train_df['sii'].values.copy()

    corr_sii = df_corr.corr(numeric_only=True)['sii']
    corr_sii = corr_sii[(corr_sii > 0.02) | (corr_sii < -0.02)]

    corr_list = corr_sii.keys().tolist()

    # corr_list.remove('sii')

    train_df_with_correlation = train_df[corr_list].copy()
    common_columns = [col for col in corr_list if col in train_df.columns and col in test_df.columns]
    common_columns.append('id')
    test_df_with_correlation = test_df[common_columns].copy()

    return train_df_with_correlation, test_df_with_correlation



def drop_sds_columns(train_df, test_df):
    def remove_sds_column(df):
        if 'SDS-SDS_Total_Raw' in df.columns:
            df.drop(['SDS-SDS_Total_Raw'], axis=1, inplace=True)
    remove_sds_column(train_df)
    remove_sds_column(test_df)
    return train_df, test_df

def drop_pciat_columns(train_df, test_df):
    def drop_pciat_columns(df):
        if 'PCIAT-PCIAT_Total' in df.columns:
            df.drop(columns='PCIAT-PCIAT_Total', inplace=True)
    drop_pciat_columns(train_df)
    drop_pciat_columns(test_df)
    return train_df, test_df

'''
If we want to compare the variance of 2 columns, we want to make sure we're normalizing at the correct time. Things like categorical integers and percentages should not be normalized. This creates a list of columns that should not be normalized.
'''

def get_features_numeric_and_non_categorical_features_from_df(df):
    data_dictionary = pd.read_csv('data_dictionary.csv')
    numeric_columns = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
    dd_fields = data_dictionary[data_dictionary['Type'] == 'categorical int']['Field'].tolist()

    # Only normalize numeric columns that are NOT categorical int
    features = [col for col in numeric_columns if col not in dd_fields and col != 'sii']
    return features

def remove_highly_correlated_columns(train_df, test_df):
    def get_features_numeric_and_non_categorical_features_from_df(df):
        data_dictionary = pd.read_csv('data_dictionary.csv')
        numeric_columns = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
        dd_fields = data_dictionary[data_dictionary['Type'] == 'categorical int']['Field'].tolist()

        # Only normalize numeric columns that are NOT categorical int
        features = [col for col in numeric_columns if col not in dd_fields and col != 'sii']
        return features
    features_to_normalize = get_features_numeric_and_non_categorical_features_from_df(train_df)

    '''
    Compare the highest correlated columns in `train_df`. If 2 features have a correlation >= 0.99, drop the one with less variance.
    '''

    def handle_highly_correlated_columns(df, threshold=0.99):
        numeric_cols = df.select_dtypes(include=['int32', 'int64', 'float64']).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Mask to avoid duplicate and self-correlations. Bascially a matrix that masks everything except the upper triangle
        upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

        # Find feature pairs with high correlation
        high_corr_pairs = [
            (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            for i, j in zip(*np.where((abs(correlation_matrix) > threshold) & upper_triangle))
        ]

        columns_to_drop = set()
        for col1, col2, corr_value in high_corr_pairs:
            # Normalize the columns ONLY if they are in features_to_normalize
            if col1 in features_to_normalize and col2 in features_to_normalize:
                normalized_col1 = (df[col1] - df[col1].mean()) / df[col1].std()
                normalized_col2 = (df[col2] - df[col2].mean()) / df[col2].std()

                var_col1 = normalized_col1.var()
                var_col2 = normalized_col2.var()
            else:
                var_col1 = df[col1].var()
                var_col2 = df[col2].var()

            # Drop the column with lesser normalized variance
            if var_col1 >= var_col2:
                # print(f"Column {col2} has a correlation of {corr_value} with {col1}")
                columns_to_drop.add(col2)
            else:
                # print(f"Column {col1} has a correlation of {corr_value} with {col2}")
                columns_to_drop.add(col1)
        return columns_to_drop

    def drop_columns_from_dataframe(df, cols=[]):
        for col in cols:
            if col in df.columns:
                # print(f"Dropping column {col}")
                df.drop(columns=col, inplace=True)

    columns_to_drop = handle_highly_correlated_columns(train_df)

    drop_columns_from_dataframe(train_df, columns_to_drop)
    drop_columns_from_dataframe(test_df, columns_to_drop)

    return train_df, test_df


    '''
    In the EDA above, there was no correlation between Season and `sii` or any other metric. Drop those columns
    '''
def drop_seasons(train_df, test_df):
    def drop_season_cols(df):
        season_cols = [col for col in df.columns if 'Season' in col]
        df = df.drop(season_cols, axis=1, inplace=True) 
    drop_season_cols(train_df)
    drop_season_cols(test_df)
    return train_df, test_df


def cap_outlier_scores_at_100(train_df, test_df):
    '''
    We should also check for outliers and replace them with `NaN`. As an example, `CGAS-CGAS_Score` has a value of 999, which is an error
    '''

    def handle_outliers(df, column, valid_min, valid_max, placeholder_value=np.nan):
        outliers = (df[column] < valid_min) | (df[column] > valid_max)
        
        # print(f"Found {outliers.sum()} outliers in column '{column}'.")
        
        df.loc[outliers, column] = placeholder_value

    handle_outliers(train_df, 'CGAS-CGAS_Score', 0, 100, np.nan)
    handle_outliers(test_df, 'CGAS-CGAS_Score', 0, 100, np.nan)
    handle_outliers(train_df, 'SDS-SDS_Total_T', 0, 100, np.nan)
    handle_outliers(test_df, 'SDS-SDS_Total_T', 0, 100, np.nan)
    return train_df, test_df

    '''
    There are a handful of records that are almost entirely null. We will remove these from the training data. However, every record has a value for`id`, `Basic_Demos-Age`, and `Basic_Demos-Sex`. Therefore, if a record has all remaining columns listes as `NaN`, it is safe to drop them from `train_df`. 
    '''

    # print(f"There are {len(train_df.columns)} columns in the training data and {len(test_df.columns)} columns in the test data.\n")
    # null_counts_per_row = train_df.isnull().sum(axis=1)
    # null_counts_distribution = null_counts_per_row.value_counts().sort_index()
    # print(f"Here are the number of null columns each record has:\n {null_counts_distribution}")

    # null_counts_per_row = train_df.isnull().sum(axis=1)
    # non_null_columns = len(train_df.columns) - 3
    # df_with_max_nulls = train_df[null_counts_per_row == non_null_columns]



def drop_rows_that_are_above_null_threshold(train_df, test_df):
    threshold = 0.1  # require at least 10% of the dataset be non-null
    row_threshold = int(threshold * train_df.shape[1])
    train_df.dropna(thresh=row_threshold, axis=0, inplace=True)

    pd.set_option('display.max_rows', 10)
    # print(f"There are {len(train_df.columns)} columns in the training data and {len(test_df.columns)} columns in the test data.\n")
    null_counts_per_row = train_df.isnull().sum(axis=1)
    null_counts_distribution = null_counts_per_row.value_counts().sort_index()
    # print(f"Here are the number of null columns each record has:\n {null_counts_distribution}")

    return train_df, test_df


def knn_to_inpute_non_categorical_data(train_df, test_df, n_neighbors=5):
    '''
    ### Use k-NN to Impute Missing Numeric, Non-categorical Data
    '''
    def knn_impute(df, n_neighbors=n_neighbors):
        features_to_impute = get_features_numeric_and_non_categorical_features_from_df(train_df)
        
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Perform k-NN imputation and ensure the result integrates with the DataFrame
        imputed_data = imputer.fit_transform(df[features_to_impute])
        df[features_to_impute] = pd.DataFrame(imputed_data, columns=features_to_impute, index=df.index)
        
        return df

    train_df = knn_impute(train_df, n_neighbors=5)
    test_df = knn_impute(test_df, n_neighbors=5)

    return train_df, test_df
    # Basic_Demos-Sex was removed from this list


def knn_to_inpute_categorical_data(train_df, test_df, n_neighbors=5):
    '''
    ### Use k-NN to Impute Missing Numeric, Categorical Data
    '''
    def knn_impute_categorical(df, categorical_columns, n_neighbors=n_neighbors):
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Need to check this because test_df doesn't have all the columns as train_df
        valid_columns = [col for col in categorical_columns if col in df.columns]

        # Fit-transform on valid columns
        imputed_data = imputer.fit_transform(df[valid_columns])
        df[valid_columns] = pd.DataFrame(imputed_data, columns=valid_columns, index=df.index)

        df[valid_columns] = df[valid_columns].round().astype(int)
        return df

    numeric_categorical_data = ['FGC-FGC_CU_Zone','FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone','FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_SRR_Zone','FGC-FGC_TL_Zone','BIA-BIA_Activity_Level_num','BIA-BIA_Frame_num','PCIAT-PCIAT_01','PCIAT-PCIAT_02','PCIAT-PCIAT_03','PCIAT-PCIAT_04','PCIAT-PCIAT_05','PCIAT-PCIAT_06','PCIAT-PCIAT_07','PCIAT-PCIAT_08','PCIAT-PCIAT_09','PCIAT-PCIAT_10','PCIAT-PCIAT_11','PCIAT-PCIAT_12','PCIAT-PCIAT_13','PCIAT-PCIAT_14','PCIAT-PCIAT_15','PCIAT-PCIAT_16','PCIAT-PCIAT_17','PCIAT-PCIAT_18','PCIAT-PCIAT_19','PCIAT-PCIAT_20','PreInt_EduHx-computerinternet_hoursday']

    train_df = knn_impute_categorical(train_df, numeric_categorical_data)
    test_df = knn_impute_categorical(test_df, numeric_categorical_data)

    # train_df.dropna(subset=['sii'], inplace=True)

    return train_df, test_df

def calculate_qwk(y1, y2):
    return cohen_kappa_score(y1, y2, weights='quadratic')



def do_train_test_split(train_df, test_df):
    X = train_df.drop(columns=['sii'])
    y = train_df['sii']
    
    # Need to drop all the PCIAT-PCIAT_# columns because they are not given in the test data
    X = X.drop(columns=[col for col in X.columns if 'PCIAT-PCIAT' in col])

    X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_full_train, X_test, y_full_train, y_test


def train_using_random_forest_classification(X_full_train, X_test, y_full_train, y_test):
    print(f"\n\nRunning train_using_random_forest_classification()\n")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full_train, y_full_train)):
        # Split train and validation data for this fold
        X_fold_train, X_fold_val = X_full_train.iloc[train_idx], X_full_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_full_train.iloc[train_idx], y_full_train.iloc[val_idx]

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_fold_train, y_fold_train)

        # Make predictions on the validation set
        y_fold_pred = model.predict(X_fold_val)

        # Evaluate performance using QWK
        qwk = calculate_qwk(y_fold_val, y_fold_pred)  # Use your calculate_qwk function
        qwk_scores.append(qwk)

        print(f"Fold {fold + 1}: QWK = {qwk:.4f}")

    # Calculate average performance across folds
    average_qwk = sum(qwk_scores) / len(qwk_scores)
    print(f"Average QWK across folds: {average_qwk:.4f}")

    y_pred = model.predict(X_test)
    print(f"\nQuadratic Weighted Kappa: {calculate_qwk(y_test, y_pred):.4f}\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))


def train_using_light_gbm(X_full_train, X_test, y_full_train, y_test):
    print(f"\n\nRunning train_using_light_gbm()\n")
    train_data = lgb.Dataset(X_full_train, label=y_full_train)

    # Set parameters for multi-class classification
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 4,  # Number of classes
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=100)

    # Predict class probabilities for each class
    y_pred_proba = model.predict(X_test)

    # Convert probabilities to class labels (e.g., by choosing the class with the highest probability)
    y_pred = y_pred_proba.argmax(axis=1)

    qwk_score = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    print(f"Quadratic Weighted Kappa: {qwk_score:.4f}")

def train_using_catboost(X_full_train, X_test, y_full_train, y_test):
    print(f"\n\nRunning train_using_catboost()\n")
    model = CatBoostClassifier(iterations=1000,
                        depth=6,
                        learning_rate=0.05,
                        loss_function='MultiClass',
                        verbose=True,
                        l2_leaf_reg=4
                        )

    possible_cat_features = ['FGC-FGC_CU_Zone','FGC-FGC_GSND_Zone','FGC-FGC_GSD_Zone','FGC-FGC_PU_Zone','FGC-FGC_SRL_Zone','FGC-FGC_SRR_Zone','FGC-FGC_TL_Zone','BIA-BIA_Activity_Level_num','BIA-BIA_Frame_num','PCIAT-PCIAT_01','PCIAT-PCIAT_02','PCIAT-PCIAT_03','PCIAT-PCIAT_04','PCIAT-PCIAT_05','PCIAT-PCIAT_06','PCIAT-PCIAT_07','PCIAT-PCIAT_08','PCIAT-PCIAT_09','PCIAT-PCIAT_10','PCIAT-PCIAT_11','PCIAT-PCIAT_12','PCIAT-PCIAT_13','PCIAT-PCIAT_14','PCIAT-PCIAT_15','PCIAT-PCIAT_16','PCIAT-PCIAT_17','PCIAT-PCIAT_18','PCIAT-PCIAT_19','PCIAT-PCIAT_20','PreInt_EduHx-computerinternet_hoursday']

    cat_features = [col for col in X_full_train.columns if col in possible_cat_features]

    # lightgbm = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6)
    # ensemble = VotingClassifier(estimators=[('catboost', catboost), ('lightgbm', lightgbm)], voting='soft')
    
    model.fit(X_full_train, y_full_train)

    # Train the model
    model.fit(X_full_train, y_full_train, cat_features=cat_features)

    # Make the prediction using the resulting model
    y_pred = model.predict(X_test)

    # Calculate the Quadratic Weighted Kappa score
    qwk_score = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    print(f"Quadratic Weighted Kappa: {qwk_score:.4f}")
    print(f"\nQuadratic Weighted Kappa: {calculate_qwk(y_test, y_pred):.4f}\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ids = X_full_train['id']
    # y_pred.to_csv('training_submission.csv', index=False)
    return model


def main():
    '''
    # Apply model to test set
    '''

    ids = test_df['id']
    X_final = test_df.drop(columns=['id'] + [col for col in test_df.columns if 'PCIAT-PCIAT' in col])
    y_pred_final = model.predict(X_final)
    # y_pred_final = y_pred_final.argmax(axis=1)

    submission = pd.DataFrame({
        'id': ids,
        'prediction': y_pred_final
    })

    # submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    print('running')
    train_df, test_df, train_ds, test_ds = load_dater()
    train_df, test_df, train_ds, test_ds = fill_in_nans_on_time_series_data(train_df, test_df, train_ds, test_ds)
    train_df, test_df = remove_low_correlation_columns(train_df, test_df)
    train_df, test_df = drop_sds_columns(train_df, test_df)
    train_df, test_df = drop_pciat_columns(train_df, test_df)
    train_df, test_df = remove_highly_correlated_columns(train_df, test_df)
    train_df, test_df = drop_seasons(train_df, test_df)
    train_df, test_df = cap_outlier_scores_at_100(train_df, test_df)
    train_df, test_df = drop_rows_that_are_above_null_threshold(train_df, test_df)
    train_df, test_df = knn_to_inpute_non_categorical_data(train_df, test_df, n_neighbors=5)
    train_df, test_df = knn_to_inpute_categorical_data(train_df, test_df, n_neighbors=5)

    train_df.dropna(subset=['sii'], inplace=True)
    X_full_train, X_test, y_full_train, y_test = do_train_test_split(train_df, test_df)

    # train_using_random_forest_classification(X_full_train, X_test, y_full_train, y_test)
    # train_using_light_gbm(X_full_train, X_test, y_full_train, y_test)
    model = train_using_catboost(X_full_train, X_test, y_full_train, y_test)
    
    # print(test_df.head())
    ids = test_df['id']
    X_final = test_df.drop(columns=['id'] + [col for col in test_df.columns if 'PCIAT-PCIAT' in col])
    print(X_final.columns)
    print(X_final.head())
    y_pred_final = model.predict(X_final)
    y_pred_final = y_pred_final.argmax(axis=1)

    submission = pd.DataFrame({
        'id': ids,
        'prediction': y_pred_final
    })
    submission.to_csv('submission.csv', index=False)



'''
0.3919
model = CatBoostClassifier(iterations=1000,
                    depth=6,
                    learning_rate=0.05,
                    loss_function='MultiClass',
                    verbose=True,
                    l2_leaf_reg=4
                    )

'''