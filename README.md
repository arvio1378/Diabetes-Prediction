# 📚 Diabetes Prediction Logistic Regression

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

Information about dataset attributes :
- Pregnancies: To express the Number of pregnancies
- Glucose: To express the Glucose level in blood
- BloodPressure: To express the Blood pressure measurement
- SkinThickness: To express the thickness of the skin
- Insulin: To express the Insulin level in blood
- BMI: To express the Body mass index
- DiabetesPedigreeFunction: To express the Diabetes percentage
- Age: To express the age
- Outcome: To express the final result 1 is Yes and 0 is No

## 🖥️ Cara Menjalankan Program
1. Clone repositori
```bash
https://github.com/arvio1378/Diabetes-Prediction-Logistic-Regression.git
cd Diabetes-Prediction-Logistic-Regression
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Jalankan program
```bash
python main.py
```

## 📈 Hasil & Evaluasi
Model dapat memberikan hasil akurasi score sebesar 0.79 sehingga dapat dikatakan bahwa model dapat bekerja dengan baik untuk memprediksi diabetes. Pasien non diabetes diprediksi lebih akurat daripada pasien diabetes yang dapat dilihat dari hasil precision dan recall yang lebih tinggi daripada pasien diabetes.

## 🏗️ Kontribusi
Dapat melakukan kontribusi kepada siapa saja. Bisa bantu untuk :
- Perbaikan model (Hyperparameter Tuning)
- Menggunakan data yang lebih besar
- Menambahkan antaramuka di web/streamlit

## 🧑‍💻 Tentang Saya
Saya sedang belajar dan membangun karir di bidang AI/ML. Projek ini adalah latihan saya untuk membangun aplikasi python sederhana. Saya ingin lebih untuk mengembangkan skill saya di bidang ini melalui projek-projek yang ada.

📫 Terhubung dengan saya di:
- Linkedin : https://www.linkedin.com/in/arvio-abe-suhendar/
- Github : https://github.com/arvio1378
