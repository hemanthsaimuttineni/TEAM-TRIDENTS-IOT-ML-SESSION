import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

def generateAi():
    # Load dataset
    data = pd.read_csv('Data.csv')

    # Features (all columns except last)
    X = data.iloc[:, :-1].values

    # Target (last column)
    y = data.iloc[:, -1].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    ai = KNeighborsClassifier(n_neighbors=5)
    ai.fit(X_train, y_train)

    # Evaluate model
    y_pred = ai.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {acc:.2f}")

    # Save trained model
    with open('ai.pkl', 'wb') as f:
        pickle.dump(ai, f)
