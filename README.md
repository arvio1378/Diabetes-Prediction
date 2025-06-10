# 📚 Diabetes-Prediction-Logistic-Regression

## 📋 Deskripsi
Projek ini adalah projek latihan saya untuk membuat aplikasi untuk memprediksi apakah pasien terdapat diabetes atau tidak menggunakan algoritma logistic regression. Untuk memprediksi diabetes menggunakan beberapa faktor seperti BMI, glukosa, insulin, dan faktor lainnya yang berkorelasi dengan diabetes.

## 🚀 Fitur
- Input berupa faktor faktor yang dibutuhkan untuk memprediksi diabetes menggunakan CLI
- Model Logistic Regression dengan scikit-learn
- Dataset disimpan dalam bentuk CSV
- Evaluasi model dengan Accuracy Score dan Classification Report
- Hasil split data dan model disimpan ke file joblib yang berbeda untuk bisa digunakan di file lain
- Validasi input pada CLI untuk mencegah kesalahan

## 🧠 Tools & Library
- Python 3.X
- Pandas
- Joblib
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn

## 📁 Struktur Folder
- Diabetes Prediction Logistic Regression/
  - data
      - diabetes.csv
  - src
      - model.pkl
      - model.ipynb
      - preprocessing.ipynb
      - data_clean.pkl
  - main.py
  - requirements.txt
  - dataDescription.txt
  - README.md

 ## 📊 Dataset
| Pregnancies | Glucose     | BloodPressure | SkinThickness | Insulin     | BMI         | DiabetesPedigreeFunction | Age         | Outcome     |
|-------------|-------------|---------------|---------------|-------------|-------------|--------------------------|-------------|-------------|
| 6           | 148         | 72            | 35            | 0           | 33.6        | 0.627                    | 50          | 1           |
| 1           | 85          | 66            | 29            | 0           | 26.6        | 0.351                    | 31          | 0           |
| 8           | 183         | 64            | 0             | 0           | 23.3        | 0.672                    | 32          | 1           |
| 1           | 89          | 66            | 23            | 94          | 28.1        | 0.167                    | 21          | 0           |
| 0           | 137         | 40            | 35            | 168         | 43.1        | 2.288                    | 33          | 1           |
| ...         | ...         | ...           | ...           | ...         | ...         | ...                      | ...         | ...         |
