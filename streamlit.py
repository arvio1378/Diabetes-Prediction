import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🧑‍⚕️",
    layout="wide"
)

# Load dataset
@st.cache_data # fungsi agar tidak selalu meload dataset dari awal
def load_data():
    df = pd.read_csv("./data/diabetes.csv")

    x = df.drop(["Outcome"], axis=1) # fitur 
    y = df["Outcome"] # target

    # Ubah nilai yang tidak seharusnya 0 menjadi NaN
    kolomNol = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[kolomNol] = df[kolomNol].replace(0, np.nan)
    # Ganti nilai NaN dengan median
    nilai_median = df[kolomNol].median()
    df[kolomNol] = df[kolomNol].fillna(nilai_median)

    # Outlier
    for col in x:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

# Navigasi sidebar
st.sidebar.title("Navigation")
# Pilihan Navigasi
pages = st.sidebar.selectbox("Menu : ", ["Profile", "Overview", "EDA", "Evaluation", "Prediction"])


# Page 1 : Profile
if pages == "Profile":
    # Judul aplikasi dan huruf italic
    st.title("Arvio Abe Suhendar")
    # Subheader
    st.subheader("Career Shifter | From Network to AI | Designing Intelligent Futures | Ready to Make an Impact in AI | Python Developer | Machine Learning Engineer | Data Scientist")
    st.markdown("---")

    # About me
    st.write("### 📝 About Me")
    st.write("👨‍💻 I'm a tech enthusiast with a strong foundation in Informatics Engineering from Universitas Gunadarma, where I developed solid analytical thinking, programming, and problem-solving skills.")
    st.write("🔧 After graduating, I began my professional journey as a Junior Network Engineer, managing enterprise network services like VPNIP, Astinet, and SIP Trunk on Huawei and Cisco platforms—handling configurations, service activations, and troubleshooting.")
    st.write("🤖 Over time, my curiosity led me to explore the world of Artificial Intelligence & Machine Learning. I've been actively upskilling through bootcamps and self-learning—covering data preprocessing, supervised & unsupervised learning, and deep learning using Python.")
    st.write("🎯 I'm now transitioning my career into AI/ML, combining my network infrastructure background with my growing expertise in data and intelligent systems. I'm particularly interested in how AI can improve systems, automate operations, and drive smarter decision-making.")
    st.write("🤝 Open to collaborations, mentorship, and new opportunities in the AI/ML space.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Education", "Experience", "Skills"])
    with tab1:
        # Pendidikan
        st.write("### 🎓 Education")
        st.write("""
        - **Bachelor of Informatics Engineering**   
        Universitas Gunadarma, 2019 - 2023, GPA 3.82/4.00
            - Built multiple applications (web & desktop) using Java, Python, and PHP in individual and team projects.
            - Built and optimized database systems
            - Learn techniques for solving mathematical problems using programming, numerical integration, and solving equations.
        - **Bootcamp AI/ML**    
        Dibimbing.id Academy, 2025 - Present
            - Mastered core concepts of Python programming including variables, data types, control structures, and functions.
            - Understanding the fundamentals of Artificial Intelligence and Machine Learning, key concepts, and applications.
            - Techniques to clean, transform, and prepare data for analysis, including handling missing data and feature scaling.
        """)
    with tab2:
        # Pengalaman
        st.write("### 💼 Experience")
        st.write("""
        - **Junior Network Engineer**   
        PT. Infomedia Nusantara, 2023 - Present
            - Astinet & VPNIP Service Management (Huawei Routers) : 
                 Handled service activation, disconnection,isolation, modification, and resumption for enterprise clients.
            - Wifi.id Service Provisioning (Cisco & WPgen) :    
                 Performed end-to-end activation and troubleshooting for public Wi-Fi services.
            - SIP Trunk International Access Control :  
                 Managed blocking and unblocking processes for international SIP trunk services to ensure secure and compliant voice connectivity
        """)
    with tab3:
        # Keterampilan
        st.write("### 🛠️ Skills")
        st.write("""
        - **Programming Languages**: Python
        - **Machine Learning**: Scikit-learn, TensorFlow, Keras
        - **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
        - **Database Management**: MySQL, PostgreSQL
        - **Networking**: Huawei Routers, Cisco Routers, WPgen
        - **Tools & Technologies**: Git, Docker, Jupyter Notebook
        - **Soft Skills**: Attention to Detail, Team Collaboration, Adaptability
        """)
    
    st.markdown("---")
    # Kontak
    st.write("### 📞 Contact Information")
    st.write("I'm currently studying and building a career in AI/ML. This project is my practice in building a simple Python application. I want to further develop my skills in this field through existing projects.")
    st.write("Feel free to contact me if you have any questions or suggestions regarding this project.")
    st.write("Email: 4rv10suhendar@gmail.com")
    st.write("LinkedIn: [Arvio Abe Suhendar](https://www.linkedin.com/in/arvio-abe-suhendar/)")
    st.write("Location: Depok, Indonesia")
    st.write("GitHub: [Arvio1378](https://github.com/arvio1378)")


# Page 2 : Overview
elif pages == "Overview":
    st.title("Overview")
    st.subheader("🧑‍⚕️ Diabetes Prediction")
    st.markdown("---")

    # Deskripsi aplikasi
    st.write("### 📋 Description")
    st.write("This project aims to create an application to predict whether a patient has diabetes using several models like decision tree, random forest, and logistic regression. The prediction uses several factors, such as BMI, glucose, insulin, and other factors correlated with diabetes.")

    # Tujuan aplikasi
    st.write("### 🎯 Objective")
    st.write("""
    - Predicting whether a patient has diabetes or not based on medical data
    - Can predict diabetes with early diagnosis with Artificial Intelligence
    """)

    # Membuat tab
    tab1, tab2, tab3 = st.tabs(["Application Features", "Tools & Library", "Folder Structure"])
    with tab1:
        # Fitur aplikasi
        st.write("### 🚀 Application Features")
        st.write("""
        - Input in the form of factors needed to predict diabetes
        - Comparing several models such as decision tree, random forest, and logistic regression
        - The dataset is saved in CSV format
        - Evaluasi model dengan Accuracy Score dan Classification Report
        - The split data and model results are saved to different joblib files so they can be used in other files.
        - Input validation to prevent errors
        """)
    with tab2:
        # Tools & Library
        st.write("### 🛠️ Tools & Library")
        st.write("""
        - Python
        - Matplotlib
        - Seaborn
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Plotly
        """)
    with tab3:
        # Struktur folder
        st.write("### 📂 Folder Structure")
        st.write("""
        ```
        ├── data
        │   └── diabetes.csv
            └── dataDescription.txt
        ├── src
        │   ├── Decision Tree.pkl
        │   ├── Logistic Regression.pkl
        │   ├── Random Forest.pkl
        │   ├── model.ipynb
        │   ├── preprocessing.ipynb
        │   └── data_clean.pkl
        ├── requirements.txt
        ├── main.py
        ├── streamlit.py
        └── README.md
        ```
        """)
    
    st.markdown("---")
    # Load dataset
    df = load_data()
    st.write("### 📊 Dataset Overview")
    st.table(df.head())
    st.write("#### Dataset : [`diabetes.csv`](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)")
    # Jumlah baris dan kolom
    st.write(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")
    # Deskripsi dataset
    st.write("### 🗒️ Dataset Description")
    st.write("""
    Information about dataset attributes :
    - **Pregnancies**: To express the Number of pregnancies
    - **Glucose**: To express the Glucose level in blood
    - **BloodPressure**: To express the Blood pressure measurement
    - **SkinThickness**: To express the thickness of the skin
    - **Insulin**: To express the Insulin level in blood
    - **BMI**: To express the Body mass index
    - **DiabetesPedigreeFunction**: To express the Diabetes percentage
    - **Age**: To express the age
    - **Outcome**: To express the final result 1 is Yes and 0 is No
    """)

    st.markdown("---")
    # Deskripsi dataset
    st.write("### 📈 Statistics Summary")
    df_desc = df.describe()
    st.table(df_desc)

# Page 3 : EDA
elif pages == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    df = load_data()
    st.write("### Dataset Summary")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        # Does Glucose level affect the possibility of having diabetes?
        fig = px.box(df, x="Outcome", y="Glucose", title="Glucose Level vs Diabetes Outcome")
        st.plotly_chart(fig)
        st.write("Patients with diabetes have higher glucose levels than those without diabetes. This suggests that blood sugar levels are a key factor in determining whether a patient has diabetes.")
    with col2:
        # What is the age distribution of patients with diabetes compared to those without?
        fig = px.histogram(df, x="Age", color="Outcome", barmode="group", title="Age Distribution by Diabetes Outcome")
        st.plotly_chart(fig)
        st.write("Diabetes is most common in people aged 30 to 60 and above. The least common age group for diabetes is between 20 and 30.")
    
    col1, col2 = st.columns(2)
    with col1:
        # What percentage of patients suffer from diabetes?
        diabetes_count = df["Outcome"].value_counts()
        labels = ["No Diabetes", "Diabetes"]
        fig = px.pie(values=diabetes_count, names=labels, title="Percentage of Diabetes Patients")
        st.plotly_chart(fig)
        st.write("As many as 34.9% had diabetes so it can be concluded that more people did not have diabetes as much as 65.1%.")
    with col2:
        # How big a role do BMI and Glucose play in predicting diabetes?
        fig = px.scatter(df, x="Glucose", y="BMI", color="Outcome", title="BMI vs Glucose by Diabetes Outcome")
        st.plotly_chart(fig)
        st.write("The graph shows that patients with high BMI and glucose values have diabetes.")
    
    left, center, right = st.columns([1, 2, 1])
    with center:
        # Correlation Matrix
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.write("The most correlated feature to predict diabetes is 'Glucose' as much as 0.49 and the lowest is 'BloodPressure' as much as 0.17. For the highest relationship is between 'SkinThickness' and 'BMI' as much as 0.56.")


# Page 4 : Evaluation
elif pages == "Evaluation":
    x_train, x_test, y_train, y_test = joblib.load("src/data_clean.pkl")
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_model.fit(x_train, y_train)
    y_pred_dt = dt_model.predict(x_test)
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(x_train, y_train)
    y_pred_lr = lr_model.predict(x_test)
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    y_pred_rf = rf_model.predict(x_test)

    # Judul
    st.title("Evaluation Model")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        # Accuracy Score
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        accuracy_df = pd.DataFrame({
            "Model" : ["Decision Tree", "Logistic Regression", "Random Forest"],
            "Accuracy" : [accuracy_dt, accuracy_lr, accuracy_rf]
        })

        st.subheader("Accuracy Score")
        st.dataframe(accuracy_df.style.format(precision=2))

    with col2:
        # Precision Score
        precision_dt = precision_score(y_test, y_pred_dt, average=None)
        precision_lr = precision_score(y_test, y_pred_lr, average=None)
        precision_rf = precision_score(y_test, y_pred_rf, average=None)

        precision_df = pd.DataFrame({
            "Model" : ["Decision Tree", "Logistic Regression", "Random Forest"],
            "Precision 0" : [precision_dt[0], precision_lr[0], precision_rf[0]],
            "Precision 1" : [precision_dt[1], precision_lr[1], precision_rf[1]]
        })

        st.subheader("Precision Score")
        st.dataframe(precision_df.style.format(precision=2))
    
    col1, col2 = st.columns(2)
    with col1:
        # Recall Score
        recall_dt = recall_score(y_test, y_pred_dt, average=None)
        recall_lr = recall_score(y_test, y_pred_lr, average=None)
        recall_rf = recall_score(y_test, y_pred_rf, average=None)

        recall_df = pd.DataFrame({
            "Model" : ["Decision Tree", "Logistic Regression", "Random Forest"],
            "Recall 0" : [recall_dt[0], recall_lr[0], recall_rf[0]],
            "Recall 1" : [recall_dt[1], recall_lr[1], recall_rf[1]]
        })

        st.subheader("Recall Score")
        st.dataframe(recall_df.style.format(precision=2))

    with col2:
        # F1 Score
        f1_dt = f1_score(y_test, y_pred_dt, average=None)
        f1_lr = f1_score(y_test, y_pred_lr, average=None)
        f1_rf = f1_score(y_test, y_pred_rf, average=None)

        f1_df = pd.DataFrame({
            "Model" : ["Decision Tree", "Logistic Regression", "Random Forest"],
            "F1 0" : [f1_dt[0], f1_lr[0], f1_rf[0]],
            "F1 1" : [f1_dt[1], f1_lr[1], f1_rf[1]]
        })

        st.subheader("F1 Score")
        st.dataframe(f1_df.style.format(precision=2))
    
    # Confusion Matrix
    st.markdown("---")
    st.title("Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        st.subheader("Confusion Matrix Decision Tree")
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_dt, annot=True, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)
    with col2:
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        st.subheader("Confusion Matrix Logistic Regression")
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_lr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    left, center, right = st.columns([1, 2, 1])
    with center:
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        st.subheader("Confusion Matrix Random Forest")
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm_rf, annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    
    # ROC-AUC Score
    st.markdown("---")
    st.title("ROC-AUC Score")

    models = {
    "Decision Tree" : dt_model,
    "Logistic Regression" : lr_model,
    "Random Forest" : rf_model
    }

    left, center, right = st.columns([1, 3, 1])
    with center:
        fig, ax = plt.subplots()
        for name, model in models.items():
            y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], 'r--', label='Random Guess')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


# Page 4 : Prediction
elif pages == "Prediction":
    st.title("Diabetes Prediction Test")
    model = joblib.load("src\Decision Tree.pkl")

    # Form Data
    with st.form("Diabetes_form"):
        # Input Data
        pregnancies = st.number_input("Pregnancies")
        glucose = st.number_input("Glucose")
        bloodPressure = st.number_input("BloodPressure")
        skinThickness = st.number_input("SkinThickness")
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        diabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
        age = st.number_input("Age")

        # button submit data
        bt_submit = st.form_submit_button("Predict")
    
    # Prediksi
    if bt_submit:
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
        input_df = pd.DataFrame([data_df])

        # jalankan prediksi
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Hasil Input data
        st.write(input_df)

        # hasil prediksi
        st.subheader("Hasil Prediksi")
        if prediction == 1:
            st.write(f"Positif Diabetes ({probability:.2%})")
        else:
            st.write(f"Negatif Diabetes ({probability:.2%})")