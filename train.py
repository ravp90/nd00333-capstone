import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

run = Run.get_context()
ws = run.experiment.Workspace

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--data', type=str)

    args = parser.parse_args()

    dataset = Dataset.get_by_name(ws, name='heartdisease')
    dataset = dataset.to_pandas_dataframe()

    scaler = StandardScaler()
    scaler.fit(dataset.drop('target', axis=1))
    scaler_features = scaler.transform(dataset.drop('target', axis=1))
    df.feat = pd.DataFrame(scaler_features, columns=dataset.columns[:-1])

    df['target'] = dataset['target'].astype(int)
    dataset = df_feat.copy()

    x = dataset.drop(columns=['target'])
    y = dataset['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y)

    run.log("Regularization Strength: ", np.float(args.C))
    run.log("Max Iterations: ", np.int(args.max_iter))

    mdoel = LogisticRegression(C = args.C, max_iter=args.max_iter)\
                    .fit(x_train, y_train)

    accuracy = model.score(x_test,y_test)

    os.makedirs('hd_outputs', exist_ok = True)
    joblib.dump(model, 'hd_outputs/model.joblib')

    run.log("Accuracy: ", np.float(accuracy))

if __name__ == '__main__':
    main()

