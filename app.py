import streamlit as st
import pickle
import os
import pandas as pd

label_encodings = {
    'Gender': {"Female": 0, "Male": 1},
    'Family history of overweight': {"No": 0, "Yes": 1},
    'High caloric food consumption': {"No": 0, "Yes": 1},
    'Smoking': {"No": 0, "Yes": 1},
    'Monitor calories': {"No": 0, "Yes": 1}
}

ordinal_encodings = {
    'Vegetable consumption frequency': {"Never": 0, "Sometimes": 1, "Always": 2},
    'Daily main meals frequency': {"Between 1-2": 0, "Three": 1, "More than Three": 2},
    'Between-meal food consumption frequency': {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    'Alcohol intake': {"I do not drink": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    'Daily water intake': {"Less than a liter": 0, "Between 1 and 2 L": 1, "More than 2 L": 2},
    'Physical exercise': {"I do not have": 0, "1 or 2 days": 1, "2 or 4 days": 2, "4 or 5 days": 3, "Almost Everyday": 4},
    'Daily device usage duration': {"0-2 hours": 0, "3-5 hours": 1, "More than 5 hours": 2},
    'Mode of transportation': {"Walking": 0, "Bike": 1, "Rickshaw": 2, "Public Transportation": 3, "Private Car": 4}
}

obesity_label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight",
    3: "Obesity_Type_I",
    4: "Obesity_Type_II",
    5: "Obesity_Type_III",
}

# ============ LOAD MODEL ============

models = {}
try:
    with open('rf_model_skenario1.pkl', 'rb') as file:
        models['Skenario 1'] = pickle.load(file)
except Exception:
    models['Skenario 1'] = None
try:
    with open('rf_model_skenario2.pkl', 'rb') as file:
        models['Skenario 2'] = pickle.load(file)
except Exception:
    models['Skenario 2'] = None


def display_local_text_result(title, file_path):
    """Display a local text result file if it exists."""
    st.subheader(title)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            st.code(file.read())
    else:
        st.warning(f"File tidak ditemukan: {file_path}")


def show_eda():
    st.header("Exploratory Data Analysis (EDA)")

    display_local_text_result("Data Info", os.path.join("hasil", "eda", "eda_info.txt"))
    local_images = [
        ("Cek Balancing", os.path.join("hasil", "eda", "cek_balancing.png"))
    ]
    
    for title, image_path in local_images:
        st.subheader(title)
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f"File tidak ditemukan: {image_path}")
    
    display_local_text_result("Keterangan balancing data", os.path.join("hasil", "eda", "cek_balancing.txt"))

def show_preprocessing():
    st.header("Preprocessing Data")

    display_local_text_result("encoding data", os.path.join("hasil", "preprocessing", "encoding.txt"))
    
    local_images = [
        ("Balancing Data", os.path.join("hasil", "preprocessing", "balancing.png")),
        ("Splitting Data", os.path.join("hasil", "preprocessing", "splitting_data.png"))
    ]

    title, image_path = local_images[0]
    st.subheader(title)
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"File tidak ditemukan: {image_path}")

    display_local_text_result("Detail balancing data (SMOTE)", os.path.join("hasil", "preprocessing", "balancing.txt"))

    title, image_path = local_images[1]
    st.subheader(title)
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"File tidak ditemukan: {image_path}")

def show_skenario_1():
    st.header("Skenario 1: Random Forest (Semua Fitur)")

    display_local_text_result("Hasil akurasi training menggunakan 10-Fold Cross Validation", os.path.join("hasil", "skenario1", "training.txt"))
    display_local_text_result("Hasil akurasi testing pada data test", os.path.join("hasil", "skenario1", "testing.txt"))    
    
    local_images = [
        ("Confusion Matrix Skenario 1", os.path.join("hasil", "skenario1", "confusion_matrix.png"))
    ]
    
    for title, image_path in local_images:
        st.subheader(title)
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f"File tidak ditemukan: {image_path}")

def show_skenario_2():
    st.header("Skenario 2: ANOVA + Incremental Feature Selection + Random Forest")
    
    display_local_text_result("Hasil Seleksi Fitur Anova", os.path.join("hasil", "skenario2", "anova.txt"))
    
    local_images = [
        ("Visualisasi Hasil anova berdaskan F-Score tertinggi", os.path.join("hasil", "skenario2", "anova.png")),
        ("Visualisasi akurasi semua subset fitur pada data training", os.path.join("hasil", "skenario2", "training_img.png")),
        ("Confusion Matrix Skenario 2", os.path.join("hasil", "skenario2", "confusion_matrix.png"))
    ]
    
    title, image_path = local_images[0]
    st.subheader(title)
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"File tidak ditemukan: {image_path}")
        
    display_local_text_result("Subset hasil implementasi IFS", os.path.join("hasil", "skenario2", "ifs.txt"))
    display_local_text_result("Hasil akurasi semua subset fitur pada data training", os.path.join("hasil", "skenario2", "training.txt"))  
    
    title, image_path = local_images[1]
    st.subheader(title)
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"File tidak ditemukan: {image_path}")
        
    display_local_text_result("Evaluasi pada data test", os.path.join("hasil", "skenario2", "testing.txt"))    
        
    title, image_path = local_images[2]
    st.subheader(title)
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"File tidak ditemukan: {image_path}")


# Define the Streamlit app
def main():
    st.title("Obesity Level Prediction System")
    
    # CSS untuk styling
    st.sidebar.markdown(
        """
        <style>
        /* Styling untuk main menu buttons */
        [data-testid="stSidebar"] button {
            width: 100% !important;
            text-align: left !important;
            border-radius: 0 !important;
            padding: 10px 12px !important;
            margin: 0 !important;
            border: none !important;
            border-bottom: 1px solid #e6e6e6 !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] button:hover {
            background: #000000 !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] button:focus,
        [data-testid="stSidebar"] button:active {
            outline: none !important;
            background: #000000 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # session untuk menu navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'predict'
    
    if 'analysis_submenu' not in st.session_state:
        st.session_state.analysis_submenu = None

    if 'analysis_menu_open' not in st.session_state:
        st.session_state.analysis_menu_open = False
    
    # ========== SIDEBAR MENU ==========
    # st.sidebar.markdown("---")
    st.sidebar.markdown("## MENU")
    # st.sidebar.markdown("---")
    
    # Tombol menu utama
    if st.sidebar.button("Prediksi", key="menu_predict", use_container_width=True):
        st.session_state.page = 'predict'
        st.session_state.analysis_submenu = None
        st.session_state.analysis_menu_open = False
        
    if st.sidebar.button("Analisis", key="menu_analysis", use_container_width=True):
        st.session_state.page = 'analysis'
        st.session_state.analysis_menu_open = not st.session_state.analysis_menu_open
    

    # Submenu Analisis
    if st.session_state.page == 'analysis' and st.session_state.analysis_menu_open:
        st.sidebar.caption("Submenu Analisis")

        if st.sidebar.button("EDA", key="submenu_eda", use_container_width=True):
            st.session_state.analysis_submenu = 'eda'

        if st.sidebar.button("Preprocessing", key="submenu_prep", use_container_width=True):
            st.session_state.analysis_submenu = 'preprocessing'

        if st.sidebar.button("Skenario 1", key="submenu_sken1", use_container_width=True):
            st.session_state.analysis_submenu = 'skenario1'

        if st.sidebar.button("Skenario 2", key="submenu_sken2", use_container_width=True):
            st.session_state.analysis_submenu = 'skenario2'
            

    # ========== PAGE CONTENT ==========
    
    if st.session_state.page == 'analysis':
        if st.session_state.analysis_submenu == 'eda':
            show_eda()
        elif st.session_state.analysis_submenu == 'preprocessing':
            show_preprocessing()
        elif st.session_state.analysis_submenu == 'skenario1':
            show_skenario_1()
        elif st.session_state.analysis_submenu == 'skenario2':
            show_skenario_2()
        else:
            st.info("Pilih menu Analisis atau prediksi dari sidebar untuk memulai")
    
    elif st.session_state.page == 'predict':
        show_prediction_page()


def show_prediction_page():
    
    model_skenario_1 = models.get('Skenario 1')
    model_skenario_2 = models.get('Skenario 2')

    if model_skenario_1 is None or model_skenario_2 is None:
        st.error("Model skenario 1 dan/atau skenario 2 belum tersedia.")
        return

    st.markdown(
        "**Keterangan Fitur:** "
        "[S1 & S2] = Digunakan di kedua skenario |"
        "[S1] = Hanya digunakan di Skenario 1"
    )

    feature_usage = pd.DataFrame([
        {"Fitur": "Gender", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Age", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Height", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Weight", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Family history of overweight", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "High caloric food consumption", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Vegetable consumption frequency", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Daily main meals frequency", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Between-meal food consumption frequency", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Smoking", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Alcohol intake", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Daily water intake", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Monitor calories", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Physical exercise", "Skenario 1": "✅", "Skenario 2": "✅"},
        {"Fitur": "Daily device usage duration", "Skenario 1": "✅", "Skenario 2": "❌"},
        {"Fitur": "Transportation mode", "Skenario 1": "✅", "Skenario 2": "✅"},
    ])

    st.subheader("Keterangan Fitur")
    st.dataframe(feature_usage, use_container_width=True)

    st.subheader("Input Data")

    def feature_title(name, usage):
        return f"{name} [{usage}]"

    # isi form data mentah(kategori) untuk prediksi
    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            gender_label = st.selectbox(feature_title("Gender", "S1"), list(label_encodings['Gender'].keys()), key="pred_gender")
            age = st.number_input(feature_title("Age", "S1"), min_value=0, step=1, key="pred_age")
            height = st.number_input(feature_title("Height (m)", "S1 & S2"), min_value=0.0, step=0.01, format="%.2f", key="pred_height")
            weight = st.number_input(feature_title("Weight (kg)", "S1 & S2"), min_value=0.0, step=0.1, format="%.1f", key="pred_weight")
            family_history_overweight_label = st.selectbox(feature_title("Family history of overweight", "S1 & S2"), list(label_encodings['Family history of overweight'].keys()), key="pred_fh")
            high_caloric_food_label = st.selectbox(feature_title("High caloric food consumption", "S1 & S2"), list(label_encodings['High caloric food consumption'].keys()), key="pred_hc")
            vegetable_consumption_label = st.selectbox(feature_title("Vegetable consumption frequency", "S1 & S2"), list(ordinal_encodings['Vegetable consumption frequency'].keys()), key="pred_vc")
            main_meals_frequency_label = st.selectbox(feature_title("Daily main meals frequency", "S1 & S2"), list(ordinal_encodings['Daily main meals frequency'].keys()), key="pred_mf")

        with col_right:
            between_meal_food_label = st.selectbox(feature_title("Between-meal food consumption frequency", "S1 & S2"), list(ordinal_encodings['Between-meal food consumption frequency'].keys()), key="pred_bmf")
            smoking_label = st.selectbox(feature_title("Smoking", "S1"), list(label_encodings['Smoking'].keys()), key="pred_smoking")
            alcohol_intake_label = st.selectbox(feature_title("Alcohol intake", "S1"), list(ordinal_encodings['Alcohol intake'].keys()), key="pred_ai")
            daily_water_intake_label = st.selectbox(feature_title("Daily water intake", "S1"), list(ordinal_encodings['Daily water intake'].keys()), key="pred_dwi")
            monitor_calories_label = st.selectbox(feature_title("Monitor calories", "S1"), list(label_encodings['Monitor calories'].keys()), key="pred_mc")
            physical_exercise_label = st.selectbox(feature_title("Physical exercise", "S1 & S2"), list(ordinal_encodings['Physical exercise'].keys()), key="pred_pe")
            device_usage_duration_label = st.selectbox(feature_title("Daily device usage duration", "S1"), list(ordinal_encodings['Daily device usage duration'].keys()), key="pred_dud")
            transportation_mode_label = st.selectbox(feature_title("Transportation mode", "S1 & S2"), list(ordinal_encodings['Mode of transportation'].keys()), key="pred_tm")

        submitted = st.form_submit_button("🔮 Prediksi Dua Skenario", use_container_width=True)

    if submitted:
        # data di encode sesuai dengan model yang digunakan
        
        # Label Encoding (binary features)
        gender = label_encodings['Gender'][gender_label]
        family_history_overweight = label_encodings['Family history of overweight'][family_history_overweight_label]
        high_caloric_food = label_encodings['High caloric food consumption'][high_caloric_food_label]
        smoking = label_encodings['Smoking'][smoking_label]
        monitor_calories = label_encodings['Monitor calories'][monitor_calories_label]
        
        # Ordinal Encoding (ordered categorical features)
        vegetable_consumption = ordinal_encodings['Vegetable consumption frequency'][vegetable_consumption_label]
        main_meals_frequency = ordinal_encodings['Daily main meals frequency'][main_meals_frequency_label]
        between_meal_food = ordinal_encodings['Between-meal food consumption frequency'][between_meal_food_label]
        alcohol_intake = ordinal_encodings['Alcohol intake'][alcohol_intake_label]
        daily_water_intake = ordinal_encodings['Daily water intake'][daily_water_intake_label]
        physical_exercise = ordinal_encodings['Physical exercise'][physical_exercise_label]
        device_usage_duration = ordinal_encodings['Daily device usage duration'][device_usage_duration_label]
        transportation_mode = ordinal_encodings['Mode of transportation'][transportation_mode_label]

        # ===== FITUR UNTUK SKENARIO 1 =====
        features_skenario_1 = [[
            gender,                              
            age,                                 
            height,                              
            weight,                              
            family_history_overweight,           
            high_caloric_food,                   
            vegetable_consumption,               
            main_meals_frequency,                
            between_meal_food,                   
            smoking,                             
            alcohol_intake,                      
            daily_water_intake,                  
            monitor_calories,                    
            physical_exercise,                   
            device_usage_duration,               
            transportation_mode                  
        ]]

        # ===== FITUR UNTUK SKENARIO 2 =====
        features_skenario_2 = [[
            height,                              
            weight,                              
            family_history_overweight,           
            high_caloric_food,                   
            vegetable_consumption,               
            main_meals_frequency,                
            between_meal_food,                   
            physical_exercise,                   
            transportation_mode
        ]]

        # ===== PREDIKSI SKENARIO 1 =====
        try:
            prediction_skenario_1 = model_skenario_1.predict(features_skenario_1)
            # Ambil prediksi pertama [0] dan convert ke int, lalu mapping ke label
            pred_label_skenario_1 = obesity_label_map.get(int(prediction_skenario_1[0]), int(prediction_skenario_1[0]))
        except Exception as error:
            st.error(f"Prediksi Skenario 1 gagal: {error}")
            return

        # ===== PREDIKSI SKENARIO 2 =====
        try:
            prediction_skenario_2 = model_skenario_2.predict(features_skenario_2)
            # Ambil prediksi pertama [0] dan convert ke int, lalu mapping ke label
            pred_label_skenario_2 = obesity_label_map.get(int(prediction_skenario_2[0]), int(prediction_skenario_2[0]))
        except Exception as error:
            st.error(f"Prediksi Skenario 2 gagal: {error}")
            return

        # ===== TAMPILAN HASIL PREDIKSI =====
        result_left, result_right = st.columns(2)
        with result_left:
            st.success(f"Skenario 1 (16 fitur): **{pred_label_skenario_1}**")
        with result_right:
            st.info(f"Skenario 2 (9 fitur): **{pred_label_skenario_2}**")

if __name__ == "__main__":
    main()