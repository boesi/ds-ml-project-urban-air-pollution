import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    df = pd.read_csv('data/Train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_year'] = df['Date'].dt.dayofyear
    df.insert(4, 'target_aqi', create_bins(df['target'], 'aqi'))
    df.insert(5, 'target_health', create_bins(df['target'], 'health'))
    return df

def create_bins(data, type):
    """
    Create categorical bins from continuous air quality data.

    Parameters
    ----------
    data : array-like
        Input array-like object to be binned. Must be 1-dimensional and 
        contain numeric data compatible with pandas.cut (int, float) 
    type : str
        Type of binning to perform:
        - 'aqi': Creates 6 air quality categories based on EPA standards
        - 'health': Creates binary health categories (Healthy/Unhealthy)

    Returns
    -------
    pandas.Series
        Categorical data with labels based on the specified type:
        For 'aqi': ['Good', 'Moderate', 'Unhealthy f. Sens. G.', 
                    'Unhealthy', 'Very Unhealthy', 'Hazardous']
        For 'health': ['Healthy', 'Unhealthy']

    Notes
    -----
    AQI categories follow EPA breakpoints:
    - Good: 0-50
    - Moderate: 51-100
    - Unhealthy for Sensitive Groups: 101-150
    - Unhealthy: 151-200
    - Very Unhealthy: 201-300
    - Hazardous: >300

    Health categories use a simple binary split at AQI 100.

    References
    ----------
    .. [1] EPA AQI Technical Documentation:
           https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
    """
    labels = get_bin_labels(type)
    if type == 'aqi':
        return pd.cut(data, bins=[0, 50, 100, 150, 200, 300, float('inf')], labels = labels)
    elif type == 'health':
        return pd.cut(data, bins=[0, 100, float('inf')], labels=labels)

def get_bin_labels(type):
    if type == 'aqi':
        return ['Good', 'Moderate', 'Unhealthy f. Sens. G.', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    elif type == 'health':
        return ['Healthy', 'Unhealthy']


def get_baseline_data(data, target_name, factorize_target=False, RSEED=42):
    base_model_df = data[[target_name, 'Place_ID', 'day_of_year']]
    base_model_enc = convert_to_categorical(base_model_df, 'Place_ID')
    X = base_model_enc.drop([target_name], axis=1)
    if factorize_target:
        y_con, y_labels = pd.factorize(data[target_name])
    else:
        y_con = base_model_enc[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y_con, test_size=0.2, random_state=RSEED)
    if factorize_target:
        return X_train, X_test, y_train, y_test, y_labels
    else: 
        return X_train, X_test, y_train, y_test

    
def convert_to_categorical(data, column_name):
    column = data.pop(column_name)
    dummies = pd.get_dummies(column, drop_first=True)
    return pd.concat([data, dummies], axis=1)


def check_regression(model, X_train, X_test, y_train, y_test, with_regression=False):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    display(Markdown(f"""
|Score|Test|Train|
|:--|--:|--:|
|MAE|{mean_absolute_error(y_test, y_pred_test):.3f}|{mean_absolute_error(y_train, y_pred_train):.3f}|
|MSE|{mean_squared_error(y_test, y_pred_test):.3f}|{mean_squared_error(y_train, y_pred_train):.3f}|
|R² Score|{r2_score(y_test, y_pred_test):.3f}|{r2_score(y_train, y_pred_train):.3f}|
"""))
    if with_regression:
        check_classification_for_regression(model, X_train, X_test, y_train, y_test, 'aqi')
        check_classification_for_regression(model, X_train, X_test, y_train, y_test, 'health')


def check_classification_for_regression(model, X_train, X_test, y_train, y_test, y_labels):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    if type(y_labels) == str:
        bin_type = y_labels
        y_test, y_labels = pd.factorize(create_bins(y_test, bin_type), sort=True)
        y_train, y_labels_train = pd.factorize(create_bins(y_train, bin_type), sort=True)
        print('y_pred: ', y_pred)
        print('bins: ', create_bins(y_pred, bin_type))
        y_pred, y_labels_pred = pd.factorize(create_bins(y_pred, bin_type), sort=True)
        y_pred_train, y_labels_pred_train = pd.factorize(create_bins(y_pred_train, bin_type), sort=True)

    print('--- Test data ---')
    print('Labels')
    print(f'\ty_labels: {y_labels}\n\ty_labels_train: {y_labels_train}\n\ty_labels_pred: {y_labels_pred}\n\ty_labels_pred_train: {y_labels_pred_train}')
    print('Unique')
    print(f'\ty_pred.unique: {np.unique(y_pred)}\n\ty_train.unique: {np.unique(y_train)}\n\ty_test.unique: {np.unique(y_test)}\n\ty_train.unique: {np.unique(y_train)}')
    # print(classification_report(y_test, y_pred, target_names=y_labels))
    print(classification_report(y_test, y_pred))
    print('--- Train data ---')
    # print(classification_report(y_train, y_pred_train, target_names=y_labels))
    print(classification_report(y_train, y_pred_train))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # cmd_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=y_labels, values_format="d", cmap=plt.cm.Blues, ax=axes[0])
    cmd_test = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, values_format="d", cmap=plt.cm.Blues, ax=axes[0])
    axes[0].set_title('Test Data')
    axes[0].set_xticks(ticks=axes[0].get_xticks(), labels=axes[0].get_xticklabels(), rotation=30, ha='right')

    cmd_test = ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, display_labels=y_labels, values_format="d", cmap=plt.cm.Greens, ax=axes[1])
    axes[1].set_title('Train Data')
    axes[1].set_xticks(ticks=axes[1].get_xticks(), labels=axes[1].get_xticklabels(), rotation=30, ha='right')


def check_classification(model, X_train, X_test, y_train, y_test, y_labels, train=True, test=True):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    if test:
        print('--- Test data ---')
        print(classification_report(y_test, y_pred, target_names=y_labels))
    if train:
        print('--- Train data ---')
        print(classification_report(y_train, y_pred_train, target_names=y_labels))

    fig, axes = plt.subplots(nrows=1, ncols=train+test, figsize=((train+test) * 6, 4))
    fig.patch.set_alpha(0.0)

    # plt.subplots returns a list, if there is more than one ax
    if not isinstance(axes, (list, np.ndarray)): axes = [axes]
    ax_index = 0
    if test:
        cmd_test = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=y_labels, values_format="d", cmap=plt.cm.Blues, ax=axes[ax_index])
        axes[ax_index].set_title('Test Data')
        axes[ax_index].set_xticks(ticks=axes[0].get_xticks(), labels=axes[0].get_xticklabels(), rotation=30, ha='right')
        ax_index += 1

    if train:
        cmd_test = ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, display_labels=y_labels, values_format="d", cmap=plt.cm.Greens, ax=axes[ax_index])
        axes[ax_index].set_title('Train Data')
        axes[ax_index].set_xticks(ticks=axes[1].get_xticks(), labels=axes[1].get_xticklabels(), rotation=30, ha='right')
    
    return fig


# Define model that selects and rename features
def select_and_rename_columns(df, target_name, debug = False, feateng=False):
    """
    Select desired features from the original DataFrame and rename them.

    Parameters:
    - df (pd.DataFrame): The original DataFrame
    - Target_name (str): The name of the desired target column
    Returns:
    - pd.DataFrame: A new DataFrame with selected and renamed features
    """


    df['windspeed'] = (df['u_component_of_wind_10m_above_ground'] ** 2 + df['v_component_of_wind_10m_above_ground'] ** 2) ** 0.5
    
    if debug:
        print(f"Debug: Added 'windspeed' column with {df['windspeed'].isnull().sum()} missing values")

    if target_name != 'target' and 'target' in df.columns:
        if debug:
            print(f"Debug: Dropping original 'target' column to avoid duplication.")
        df.drop(columns=['target'], inplace=True)
    
    # Rename columns as decided
    rename_dict = {
            target_name: 'target',
            'specific_humidity_2m_above_ground': 'specific_humidity',
            'temperature_2m_above_ground': 'temperature',
            'L3_NO2_NO2_column_number_density': 'NO2_conc',
            'L3_O3_O3_column_number_density': 'O3_conc',
            'L3_CO_CO_column_number_density': 'CO_conc',
            'L3_HCHO_tropospheric_HCHO_column_number_density': 'FA_conc',
            'L3_CLOUD_cloud_fraction': 'cloud_coverage',
            'L3_CLOUD_cloud_optical_depth': 'cloud_density',
            'L3_AER_AI_absorbing_aerosol_index': 'AAI',
            'L3_SO2_SO2_column_number_density': 'SO2_conc'
            }
    
    df.rename(columns=rename_dict, inplace=True)

    if debug:
        print(f"Debug: Renamed columns: {list(rename_dict.values())}")
    
    # Check if values are within documented ranges and assign NaN if not
    columns_to_check = ['NO2_conc', 'O3_conc','CO_conc', 'FA_conc', 'cloud_density', 'AAI', 'SO2_conc']
    ranges = {
        'NO2_conc': (-0.00051, 0.0192),
        'O3_conc': (0.025, 0.3048),
        'CO_conc': ( -34.43, 5.71),
        'FA_conc': (-0.0172,  0.0074),
        'cloud_density': (1, 250),
        'AAI': (-21, 39),
        'SO2_conc': (-0.4051, 0.2079)
    }
    for col in columns_to_check:
        lb, ub = ranges[col]
        old_values = df[col].copy()
        df[col] = df[col].where((df[col] >= lb) & (df[col] <= ub), np.nan)
        changed_values = old_values != df[col]    
        if debug and changed_values.any():
            changed_rows = df[changed_values]
            print(f"Debug: Changed values in column '{col}':")
            print(changed_rows[[col]])  # Print only the changed rows for clarity
    
    if feateng:
        """
        Perform feature engineering on Dataframe.
        Has to be performed before 'select_and_rename_columns' functions.

        Input: pd.DataFrame
        Output: DataFrame with transformed features
        """

        gas_map = {
        'NO2': 'NO2_conc',
        'O3':  'O3_conc',
        'HCHO': 'FA_conc',
        'SO2': 'SO2_conc'
        }

        for gas, gas_col in gas_map.items():
            cloud_col = f"L3_{gas}_cloud_fraction"
            new_col = f"{gas_col}_weighted"
            df[new_col] = df[gas_col] * (1.0 - df[cloud_col])

        columns_to_keep_feat = [
            'target', 
            "temperature", 
            "specific_humidity", 
            'NO2_conc_weighted',
            'O3_conc_weighted', 
            'FA_conc_weighted', 
            'SO2_conc_weighted',
            "cloud_coverage", 
            "cloud_density",
            "AAI",
            'CO_conc',
            'windspeed'            
            ]
        df_selected = df[columns_to_keep_feat].copy()
    

    else:
    # Select the specified columns

        columns_to_keep = [
            'target', 
            "temperature", 
            "specific_humidity", 
            "NO2_conc",
            "O3_conc",
            "CO_conc",
            "FA_conc",
            "cloud_coverage", 
            "cloud_density",
            "AAI", 
            "SO2_conc",
            'windspeed'
        ]
        df_selected = df[columns_to_keep].copy()

        if debug:
            print(f"Debug: Selected columns: {df_selected.columns.tolist()}")
    
    return df_selected


