# Core Package
import streamlit as st
import os
import joblib

# EDA Packages
import pandas as pd
import numpy as np

# Data Visualization Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to get relative paths
def get_relative_path(relative_path):
    return os.path.join(BASE_DIR, relative_path).replace('/', '\\')

# Evrensel dosya yolu oluşturma
data_path = get_relative_path('data/car.data')

buying_image_paths = {
    'low': get_relative_path('images\\low_buying.png'),
    'med': get_relative_path('images\\medium_buying.png'),
    'high': get_relative_path('images\\high_buying.jpg'),
    'vhigh': get_relative_path('images\\vhigh_buying.png')
}

maint_image_paths = {
    'low': get_relative_path('images\\low_maint.png'),
    'med': get_relative_path('images\\medium_maint.png'),
    'high': get_relative_path('images\\high_maint.png'),
    'vhigh': get_relative_path('images\\vhigh_maint.png')
}

doors_image_paths = {
    '2': get_relative_path('images\\2_doors.png'),
    '3': get_relative_path('images\\3_doors.png'),
    '4': get_relative_path('images\\4_doors.png'),
    '5more': get_relative_path('images\\5more_doors.png')
}

person_image_paths = {
    '2': get_relative_path('images\\2_persons.png'),
    '4': get_relative_path('images\\4_persons.png'),
    'more': get_relative_path('images\\more_persons.png')
}

lug_boot_image_paths = {
    'small': get_relative_path('images\\small_lug_boot.png'),
    'med': get_relative_path('images\\med_lug_boot.png'),
    'big': get_relative_path('images\\big_lug_boot.png')
}

safety_image_paths = {
    'low': get_relative_path('images\\low_safety.png'),
    'med': get_relative_path('images\\med_safety.png'),
    'high': get_relative_path('images\\high_safety.png')
}






@st.cache_data
def load_data(dataset):
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(dataset,names= cols)
    return df

def get_image_html(label, image_paths):
    image_path = image_paths[label]
    return f'<img src="{image_path}" width="50" style="vertical-align:middle;"> {label}'

def display_image_with_label(label, image_paths):
    image_path = image_paths[label]
    st.image(image_path, width=30, caption=label)

buying_label = { 'low': 0, 'med': 1, 'high': 2,'vhigh': 3}
maint_label = { 'low': 0, 'med': 1, 'high': 2,'vhigh': 3}
doors_label = {'2': 0, '3': 1, '4': 2, '5more': 3}
person_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'med': 1, 'big': 2}
safety_label = {'low': 0, 'med': 1, 'high': 2}
class_label = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}


# Get the Keys
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# Find the Key From Dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
        
def map_class_label(label):
    if label == "acc":
        return "acceptable"
    elif label == "good":
        return "good"
    elif label == "vgood":
        return "very good"
    elif label == "unacc":
        return "unacceptable"
    else:
        return None


def get_encoded_class_label(val, class_label):
    for key, value in class_label.items():
        if val == key:
            return value

def get_original_class_label(val, class_label):
    for key, value in class_label.items():
        if val == value:
            return key

# Load Model
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    """Car Evaluation Machine Learning App"""

    st.title("Car Evaluation Machine Learning App")
    st.subheader("Built with Streamlit")

    # Menu
    menu = ["EDA","Prediction","About"]
    choices = st.sidebar.selectbox("Select Activities",menu)

    if choices == 'EDA':
        st.subheader("EDA")

        data = load_data(data_path)
        st.dataframe(data.head(10))

        if st.checkbox("Show Summary"):
            st.write(data.describe())
        
        if st.checkbox("Show Shape"):
            st.write(data.shape)

        if st.checkbox("Value Count Plot"):
            st.write(data['class'].value_counts().plot(kind= 'bar'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        
        if st.checkbox("Pie Chart"):
            st.write(data['class'].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()


    if choices == 'Prediction':
        st.subheader("Prediction")

        buying_options = {label: buying_image_paths[label] for label in buying_label.keys()}
        buying = st.selectbox("Select Buying Level", options=list(buying_options.keys()))
        if buying:
            st.write("Selected Buying Level:")
            st.image(buying_options[buying], caption=buying, width=100)
        

        maint_options = {label: maint_image_paths[label] for label in maint_label.keys()}
        maint = st.selectbox("Select Maintenance Level", options=list(maint_options.keys()))
        if maint:
            st.write("Selected Maintenance Level:")
            st.image(maint_options[maint], caption=maint, width=100)

        doors_options = {label: doors_image_paths[label] for label in doors_label.keys()}
        doors = st.selectbox("Select Number of Doors", options=list(doors_options.keys()))
        if doors:
            st.write("Selected Number of Doors:")
            st.image(doors_options[doors], caption=doors, width=100)

        persons_options = {label: person_image_paths[label] for label in person_label.keys()}
        persons = st.selectbox("Select Number of Persons", options=list(persons_options.keys()))
        if persons:
            st.write("Selected Number of Persons:")
            st.image(persons_options[persons], caption=persons, width=100)
        
        lug_boot_options = {label: lug_boot_image_paths[label] for label in lug_boot_label.keys()}
        lug_boot = st.selectbox("Select Lug Boot", options=list(lug_boot_options.keys()))
        if lug_boot:
            st.write("Selected Lug Boot:")
            st.image(lug_boot_options[lug_boot], caption=lug_boot, width=100)

        safety_options = {label: safety_image_paths[label] for label in safety_label.keys()}
        safety = st.selectbox("Select Safety Level", options=list(safety_options.keys()))
        if safety:
            st.write("Selected Safety Level:")
            st.image(safety_options[safety], caption=safety, width=100)


        # Encoding
        
        v_buying = get_value(buying, buying_label) 
        v_maint = get_value(maint, maint_label)
        v_doors = get_value (doors, doors_label)
        v_persons = get_value(persons, person_label)
        v_lug_boot = get_value(lug_boot,lug_boot_label) 
        v_safety = get_value(safety, safety_label)

        pretty_data = {
        "buying" : buying,
        "maint": maint,
        "doors": doors,
        "persons" : persons, 
        "lug_boot":lug_boot, 
        "safety": safety,
        }

        st.subheader("Options Selected")
        st.json(pretty_data)

        st.subheader("Data Encoded as")
        sample_data = [v_buying,v_maint,v_doors,v_persons,v_lug_boot,v_safety]
        st.write(sample_data)

        prep_data = np.array(sample_data).reshape(1,-1)


        if st.button("Evaluate"):
            model_files = {
                "LogisticRegression": get_relative_path('models/LogisticRegression.pkl'),
                "NaiveBayes": get_relative_path('models/NaiveBayes.pkl'),
                "MLPClassifier": get_relative_path('models/MLPClassifier.pkl')
            }

            predictions = {}

            for model_name, model_file in model_files.items():
                predictor = load_prediction_model(model_file)
                prediction = predictor.predict(prep_data)
                predictions[model_name] = get_key(prediction[0], class_label)

            st.subheader("Model Predictions")
            for model_name, prediction in predictions.items():
                st.write(f"**{model_name} Prediction:** {prediction} ({map_class_label(prediction)})")
                st.write("\n")  # Add a new line for better readability

            # Fetch the actual class from the dataset
            data = load_data(data_path)
            actual_class = data[(data['buying'] == buying) & 
                                (data['maint'] == maint) & 
                                (data['doors'] == doors) & 
                                (data['persons'] == persons) & 
                                (data['lug_boot'] == lug_boot) & 
                                (data['safety'] == safety)]['class'].values[0]
            
            st.subheader("Actual Evaluation from Dataset")
            st.write(actual_class)



    if choices == 'About':
        st.subheader("About")
        st.write("""
        ## Car Evaluation Machine Learning App
        
        Bu uygulama, kullanıcıların belirli özelliklere sahip bir arabayı değerlendirmesini sağlar. 
        Uygulama, farklı makine öğrenimi modellerini kullanarak araba verilerini analiz eder ve 
        tahminlerde bulunur. Kullanılan modeller arasında Logistic Regression, Naive Bayes ve MLP 
        Classifier bulunmaktadır.

        ### Proje Adımları:
        1. **Veri Hazırlama:** `model_trains.py` dosyasında, araba verileri yüklenir ve etiketlenir.
        2. **Model Eğitimi:** Logistic Regression, Naive Bayes ve MLP Classifier modelleri eğitilir.
        3. **Model Kaydetme:** Eğitilen modeller dosya olarak kaydedilir.
        4. **Tahmin:** `app.py` dosyasında, kullanıcıdan alınan verilerle modeller kullanılarak tahmin yapılır ve sonuçlar gösterilir.

        ### Amaç:
        Bu projede, farklı makine öğrenimi algoritmalarının araba değerlendirme problemini nasıl çözdüğünü 
        göstermek ve kullanıcıların kolayca tahminler yapabilmesini sağlamaktır.
        
        ### Proje Geliştiricileri:
        İsmail Güler - 210229052\n
        Faruk Siner - 210229012
        
        
        Veri seti linki
        https://archive.ics.uci.edu/dataset/19/car+evaluation
        
        
        
    """)
    



if __name__ == '__main__':
    main()