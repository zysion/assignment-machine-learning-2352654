import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data import read_data


X_train, X_val, y_train, y_val = read_data()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

models = {
    "Model A": MLPClassifier(hidden_layer_sizes=(10,), max_iter=561, random_state=31),
    "Model B": MLPClassifier(hidden_layer_sizes=(200,), max_iter=561, random_state=31)
}

for name, model in models.items():
    print(f"--- {name} ---")
    start_time = time.time()
    
    y_train_flat = y_train.values.ravel()
 
    model.fit(X_train_scaled, y_train_flat)
    runtime = time.time() - start_time
    
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"runtime : {runtime:.4f} giây")
    print(f"Training Score : {train_acc:.4f}")
    print(f"Validation Score: {val_acc:.4f}\n")