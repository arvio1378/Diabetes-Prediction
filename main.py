import joblib
import pandas as pd

model = joblib.load("src\Decision Tree.pkl")

def predict():
    # ambil input
    data_df = {
            "Pregnancies" : pregnancies,
            "Glucose" : glucose,
            "BloodPressure" : bloodPressure,
            "SkinThickness" : skinThickness,
            "Insulin" : insulin,
            "BMI" : bmi,
            "DiabetesPedigreeFunction" : diabetesPedigreeFunction,
            "Age" : age
        }

    # ubah ke dataframe
    df = pd.DataFrame([data_df])

    # prediksi model dan output
    predict = model.predict(df)
    print (f"Diabetes : {"YES" if predict[0] == 1 else "NO"}")

while True:
    print("="*40)
    print("PREDIKSI DIABETES DENGAN AI")
    print("="*40)

    try:
        # input
        pregnancies = int(input("Jumlah kehamilan (Pregnancies) : "))
        glucose = int(input("Glukosa (Glucose) : "))
        bloodPressure = int(input("Tekanan darah (BloodPressure) : "))
        skinThickness = int(input("Ketebalan kulit (SkinThickness) : "))
        insulin = int(input("Insulin : "))
        bmi = float(input("Indeks Massa Tubuh (BMI) : "))
        diabetesPedigreeFunction = float(input("Riwayat Genetik Diabetes (DiabetesPedigreeFunction) : "))
        age = int(input("Usia : "))

        # jalankan program
        predict()

        # keluar program
        exit = input("Keluar (y/n) ? ")
        if exit == "y":
            break
        elif exit == "n":
            continue
        else:
            print("Pilihan salah !!")
    
    except ValueError:
        print("Masukkan input yang sesuai !!")