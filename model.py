import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocess_data(df):
    """Preprocesses the raw cyclone data."""
    data = df.copy()
    data.dropna(inplace=True)
    
    X = data.drop("Cyclone", axis=1)
    y = data["Cyclone"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'original_size': len(df),
        'cleaned_size': len(data)
    }

def train_knn(processed_data, k_value):
    """Trains a K-Nearest Neighbors model."""
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(processed_data['X_train'], processed_data['y_train'])
    y_pred = knn.predict(processed_data['X_test'])
    
    accuracy = accuracy_score(processed_data['y_test'], y_pred)
    precision = precision_score(processed_data['y_test'], y_pred)
    recall = recall_score(processed_data['y_test'], y_pred)
    f1 = f1_score(processed_data['y_test'], y_pred)
    cm = confusion_matrix(processed_data['y_test'], y_pred)
    
    return knn, {'acc': accuracy, 'prec': precision, 'recall': recall, 'f1': f1, 'cm': cm}

def train_decision_tree(processed_data, max_depth, min_samples_leaf):
    """Trains a Decision Tree model."""
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    dt.fit(processed_data['X_train'], processed_data['y_train'])
    y_pred = dt.predict(processed_data['X_test'])
    
    accuracy = accuracy_score(processed_data['y_test'], y_pred)
    precision = precision_score(processed_data['y_test'], y_pred)
    recall = recall_score(processed_data['y_test'], y_pred)
    f1 = f1_score(processed_data['y_test'], y_pred)
    cm = confusion_matrix(processed_data['y_test'], y_pred)
    
    return dt, {'acc': accuracy, 'prec': precision, 'recall': recall, 'f1': f1, 'cm': cm}

def train_random_forest(processed_data, n_estimators, max_depth):
    """Trains a Random Forest model."""
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(processed_data['X_train'], processed_data['y_train'])
    y_pred = rf.predict(processed_data['X_test'])
    
    accuracy = accuracy_score(processed_data['y_test'], y_pred)
    precision = precision_score(processed_data['y_test'], y_pred)
    recall = recall_score(processed_data['y_test'], y_pred)
    f1 = f1_score(processed_data['y_test'], y_pred)
    cm = confusion_matrix(processed_data['y_test'], y_pred)
    
    return rf, {'acc': accuracy, 'prec': precision, 'recall': recall, 'f1': f1, 'cm': cm}

def train_svm(processed_data, C, kernel):
    """Trains a Support Vector Machine model."""
    svm = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    svm.fit(processed_data['X_train'], processed_data['y_train'])
    y_pred = svm.predict(processed_data['X_test'])
    
    accuracy = accuracy_score(processed_data['y_test'], y_pred)
    precision = precision_score(processed_data['y_test'], y_pred)
    recall = recall_score(processed_data['y_test'], y_pred)
    f1 = f1_score(processed_data['y_test'], y_pred)
    cm = confusion_matrix(processed_data['y_test'], y_pred)
    
    return svm, {'acc': accuracy, 'prec': precision, 'recall': recall, 'f1': f1, 'cm': cm}
