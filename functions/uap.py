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
import matplotlib.pyplot as plt
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

# Define model that selects and rename features
def select_and_rename_columns(df, target_name):
    """
    Select desired features from the original DataFrame and rename them.

    Parameters:
    - df (pd.DataFrame): The original DataFrame
    - Target_name (str): The name of the desired target column
    Returns:
    - pd.DataFrame: A new DataFrame with selected and renamed features
    """
    # Select the specified columns

    columns_to_keep = [target_name, "L3_NO2_NO2_column_number_density", "L3_O3_O3_column_number_density", "L3_CO_CO_column_number_density",
                       "L3_HCHO_tropospheric_HCHO_column_number_density", "L3_CLOUD_cloud_fraction", "L3_CLOUD_cloud_optical_depth",
                       "L3_AER_AI_absorbing_aerosol_index", "L3_SO2_SO2_column_number_density"]
    df_selected = df[columns_to_keep].copy()
    
    # Rename columns as decided
    rename_dict = {
    target_name: 'target',
    'L3_NO2_NO2_column_number_density': 'NO2_conc',
    'L3_O3_O3_column_number_density': 'O3_conc',
    'L3_CO_CO_column_number_density': 'CO_conc',
    'L3_HCHO_tropospheric_HCHO_column_number_density': 'FA_conc',
    'L3_CLOUD_cloud_fraction': 'could_coverage',
    'L3_CLOUD_cloud_optical_depth': 'could_density',
    'L3_AER_AI_absorbing_aerosol_index': 'AAI',
    'L3_SO2_SO2_column_number_density': 'SO2_conc'  
                    }
    df_selected.rename(columns=rename_dict, inplace=True)
    
    return df_selected
