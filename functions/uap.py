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