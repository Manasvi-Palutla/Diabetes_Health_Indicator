import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Diabetes_012', axis=1).values
    y = data['Diabetes_012'].values

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
    y_resampled = torch.tensor(y_resampled, dtype=torch.long)

    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test