import pandas as pd

def get_data():
    df = pd.read_csv('data/Train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_year'] = df['Date'].dt.dayofyear
    df.insert(4, 'target_aqi', 
            pd.cut(df['target'], 
                    bins=[0, 50, 100, 150, 200, 300, float('inf')],
                    labels = [0, 1, 2, 3, 4, 5,]
                #     labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
            )
    )
    df.insert(5, 'target_health', 
            pd.cut(df['target'], bins=[0, 100, float('inf')], labels=[0, 1])
        #     pd.cut(df['target'], bins=[0, 100, float('inf')], labels=['Healthy', 'Unhealthy'])
    )
    return df
    
def convert_to_categorical(data, column_name):
    column = data.pop(column_name)
    dummies = pd.get_dummies(column, drop_first=True)
    return pd.concat([data, dummies], axis=1)
    

from IPython.display import display, Markdown, Latex
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score


def check_regression(model, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    display(Markdown(f"""
|Score|Test|Train|
|:--|--:|--:|
|MAE|{mean_absolute_error(y_test, y_pred_test):.2f}|{mean_absolute_error(y_train, y_pred_train):.2f}|
|MSE|{mean_squared_error(y_test, y_pred_test):.2f}|{mean_squared_error(y_train, y_pred_train):.2f}|
|R² Score|{r2_score(y_test, y_pred_test):.2f}|{r2_score(y_train, y_pred_train):.2f}|
"""))

from sklearn.metrics import confusion_matrix
import matplot.pyplot as plt
import seaborn as sns

def check_classification(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax[0]);
    sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt="d", ax=ax[1]);

    # # Print accuracy of our model
    # print(f"Accuracy {model_name}: {round(accuracy_score(y_actual, y_pred), 2)}")
    # print("--------"*10)

    # # Print classification report of our model
    # print(classification_report(y_actual, y_pred))
    # print("--------"*10)
    # plt.showpass

    print('--- Test data ---')
    print(classification_report(y_test, y_pred, target_names=y_labels))
    print('--- Train data ---')
    print(classification_report(y_train, y_pred_train, target_names=y_labels))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].set_title('Test Data')
    cmd_test = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=y_labels, values_format="d", cmap=plt.cm.Blues, ax=axes[0])
    plt.xticks(rotation=30, ha='right')

    axes[1].set_title('Train Data')
    cmd_test = ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, display_labels=y_labels, values_format="d", cmap=plt.cm.Greens, ax=axes[1])

    plt.xticks(rotation=30, ha='right')

# Define model that selects and rename features
def select_and_rename_columns(df, target_name, debug = False):
    """
    Select desired features from the original DataFrame and rename them.

    Parameters:
    - df (pd.DataFrame): The original DataFrame
    - Target_name (str): The name of the desired target column
    Returns:
    - pd.DataFrame: A new DataFrame with selected and renamed features
    """
    # Select the specified columns

    columns_to_keep = [target_name, "temperature_2m_above_ground", "specific_humidity_2m_above_ground", "L3_NO2_NO2_column_number_density", "L3_O3_O3_column_number_density", "L3_CO_CO_column_number_density",
                       "L3_HCHO_tropospheric_HCHO_column_number_density", "L3_CLOUD_cloud_fraction", "L3_CLOUD_cloud_optical_depth",
                       "L3_AER_AI_absorbing_aerosol_index", "L3_SO2_SO2_column_number_density"]
    df_selected = df[columns_to_keep].copy()

    if debug:
        print(f"Debug: Selected columns: {df_selected.columns.tolist()}")

    df_selected['windspeed'] = (df['u_component_of_wind_10m_above_ground'] ** 2 + df['v_component_of_wind_10m_above_ground'] ** 2) ** 0.5
    
    if debug:
        print(f"Debug: Added 'windspeed' column with {df_selected['windspeed'].isnull().sum()} missing values")


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
    df_selected.rename(columns=rename_dict, inplace=True)

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
        old_values = df_selected[col].copy()
        df_selected[col] = df_selected[col].where((df_selected[col] >= lb) & (df_selected[col] <= ub), np.nan)
        changed_values = old_values != df_selected[col]    
        if debug and changed_values.any():
            changed_rows = df_selected[changed_values]
            print(f"Debug: Changed values in column '{col}':")
            print(changed_rows[[col]])  # Print only the changed rows for clarity
    

    return df_selected