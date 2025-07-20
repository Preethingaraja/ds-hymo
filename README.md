# ds-hymo

**Hybrid Machine Learning Model Deployment using Streamlit**

[Live Demo](https://ds-hymo.streamlit.app/)

---

##  Overview

**ds-hymo** is an interactive web application built using Streamlit that enables users to:
- Upload their own datasets (CSV/XLSX)
- Preprocess data with automatic label encoding
- Select independent and dependent variables
- Perform **Classification**, **Regression**, or **Clustering**
- Visualize results with **confusion matrices**, **scatter plots**, and **K-Means clusters**

This tool is ideal for beginners, data enthusiasts, and educators looking to explore machine learning workflows in a no-code or low-code environment.

---

## Features

-  Upload `.csv` or `.xlsx` datasets
-  Label Encoding for categorical columns
-  ML task selection: `Regression`, `Classification`, `Clustering`
-  Algorithms supported:
  - **Regression:** Linear Regression, Decision Tree, Random Forest
  - **Classification:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, KMeans
  - **Clustering:** KMeans with center visualization
-  Model evaluation:
  - Regression: Mean Squared Error + Scatter plot
  - Classification: Accuracy Score + Confusion Matrix (heatmap)
-  Scalable test/train split with real-time interaction

---

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## Project Structure
 ds-hymo/
│
├── pro.py # Main Streamlit app
├── requirements.txt # Required Python libraries
└── README.md # Project overview and usage guide


---

## Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/Preethingaraja/ds-hymo.git
cd ds-hymo
pip install -r requirements.txt


Run the app locally:

streamlit run pro.py

##Sample Use-Cases
Educators: Demonstrate ML algorithms in real-time to students.
Beginners: Practice model training and evaluation with your own datasets.
Analysts: Get quick insights from data with visualization and automated ML.

License
This project is open-source and available under the MIT License.

Acknowledgements
Developed by Preethinga Raja as part of hybrid ML application deployment learning initiative.

