import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Data Science Deployment Using Python")
st.header("Load Your Dataset")
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write('Dataset Loaded Successfully')
        st.write(df)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            st.write('Dataset Loaded Successfully')
        except Exception as e:
            st.error(f"An error occurred: {e}")

    label_encoders = {}
    for columns in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[columns] = le.fit_transform(df[columns])
        label_encoders[columns] = le

    st.write("Encoded successfully")
    st.write(df)

    st.header('Select Independent and Dependent Variables')
    all_columns = df.columns.tolist()
    independent_vars = st.multiselect('Select Independent Variable', all_columns)
    dependent_vars = st.selectbox('Select Dependent Variable', all_columns)

    if independent_vars and dependent_vars:
        X = df[independent_vars]
        y = df[dependent_vars]

        st.header("Split Data into Train and Test Sets")
        test_size = st.slider('Test Size (as percentage)', min_value=0.1, max_value=0.5, step=0.1, value=0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

        st.header("Choose Machine Learning Task")
        task = st.selectbox("Select Task", ["Regression", "Classification"])

        st.header("Choose Algorithm")
        model = None

        if task == "Regression":
            algorithm = st.selectbox("Select Algorithm", ['Linear Regression', 'Decision Tree', 'Random Forest'])
            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Decision Tree":
                model = DecisionTreeRegressor()
            elif algorithm == "Random Forest":
                model = RandomForestRegressor()

        elif task == "Classification":
            algorithm = st.selectbox("Select Algorithm", ['Logistic Regression', 'Decision Tree', 'Random Forest',
                                                          'Support Vector Machine', 'K Nearest Neighbors', 'K Means Clustering'])
            if algorithm == 'Logistic Regression':
                model = LogisticRegression()
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier()
            elif algorithm == 'Random Forest':
                model = RandomForestClassifier()
            elif algorithm == 'Support Vector Machine':
                model = SVC(probability=True)
            elif algorithm == 'K Nearest Neighbors':
                model = KNeighborsClassifier()
            elif algorithm == "K Means Clustering":
                model = KMeans(n_clusters=2)
                y_train = None 

        st.header("Train and Predict")

        if model is not None:
            if y_train is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task == 'Regression':
                    st.write('Mean Squared Error', mean_squared_error(y_test, y_pred))
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title(f'{algorithm} - Actual vs Predicted')
                    st.pyplot(fig)

                else:
                    st.write('Accuracy Score', accuracy_score(y_test, y_pred))
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f'{algorithm} - Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

            else:
                model.fit(X_train)
                y_pred = model.predict(X_test)
                st.write('Cluster Centers', model.cluster_centers_)
                if X_test.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
                    centers = model.cluster_centers_
                    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X')
                    ax.set_title('K-Means Clustering Results')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    st.pyplot(fig)
                else:
                    st.write("Clustering visualization skipped (need at least 2 features).")
        else:
            st.write('Please Choose a Valid algorithm to train and predict')
