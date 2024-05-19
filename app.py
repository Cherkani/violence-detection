import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import pymongo
import os
from datetime import datetime
from IPython.display import Video
from twilio.rest import Client
import pandas as pd
import base64
import plotly.express as px
import hashlib
import streamlit_option_menu as option_menu  # Importing the option_menu
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit.components.v1 as components
import requests
st.set_page_config(
    page_title="DetecterViolence",
    page_icon="icon.png",
    layout="wide",
)

# Fonction pour obtenir le modèle de prédiction
@st.cache_data()
def get_predictor_model():
    from model import Model
    model = Model()
    return model

model = get_predictor_model()

def process_video(input_video_path: str, frame_skip=5):
    cap = cv2.VideoCapture(input_video_path)
    frameST = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ne traiter que chaque nth frame
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = model.predict(frame)
            label = prediction['label']
            conf = prediction['confidence']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.putText(frame, label.title(), 
                                (0, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2)
            frameST.image(frame, channels="BGR")
        
        frame_count += 1

    cap.release()


def process_realtime():
    # Votre Twilio account SID et Twilio Auth Token
    account_sid = 'AC37bf3f09232d521ad6fb12ee91af071b'
    auth_token = '0d1e96e3a6a6e3b4678addd9ab6cd4c1'
    client = Client(account_sid, auth_token)

    cap = cv2.VideoCapture(0)
    frameST = st.empty()

    # Ajouter un bouton d'arrêt
    stop_button = st.button('Arrêter le traitement en temps réel', type="primary")

    # Définir un seuil de confiance
    # Initialiser les variables
    recording = False
    out = None
    grace_period = 30  # nombre d'images à enregistrer après l'arrêt de la violence
    grace_counter = 0

    while True:
        # Sortir de la boucle si le bouton d'arrêt est cliqué
        if stop_button:
            st.write('Traitement en temps réel arrêté.')
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convertir l'image en RGB et prédire
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame_rgb)
        label = prediction['label']
        conf = prediction['confidence']

        # Commencer l'enregistrement si de la violence est détectée et que la confiance est supérieure au seuil
        if label.lower() in ['violence', 'fire']:
            if not recording:
                recording = True
                now = datetime.now()
                date_string = now.strftime("%Y-%m-%d")
                time_string = now.strftime("%H-%M-%S")
                directory = f"recordings/{date_string}"
                os.makedirs(directory, exist_ok=True)
                filename = f"{directory}/{label}_{time_string}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, 7, (640, 480))

            out.write(frame)
            grace_counter = 0  # réinitialiser le compteur à chaque fois que la violence est détectée

        # Arrêter l'enregistrement si la violence n'est plus détectée et que la période de grâce est passée
        elif recording:
            if grace_counter < grace_period:
                grace_counter += 1
                out.write(frame)
            else:
                recording = False
                grace_counter = 0
                out.release()
                # Convertir la vidéo au format correct
                output_filename = filename.split('.')[0] + '_converted.mp4'
                print(f"Conversion de {filename} en {output_filename}")  # Statement de print avant la commande
                os.system(f'ffmpeg -i {filename} -vcodec libx264 {output_filename}')
                print("Conversion terminée")  # Statement de print après la commande
                # Supprimer la vidéo originale
                os.remove(filename)
                url = "https://seddam.mninou.com/api/bebe"
                data = {"text": "Violence detected succesfully at Violence Detector at " + now.strftime("%d-%m-%Y") + "  " +  time_string
                }
                response = requests.post(url,json=data)
                print(response)

        # Convertir l'image en BGR et ajouter du texte
        frame = cv2.putText(frame, label.title(), 
                            (0, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255), 2)
        frameST.image(frame, channels="BGR")
        time.sleep(0.01)  # contrôler le taux d'images

    cap.release()

def login_page_css():
    with open('style.css', 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Application principale
def main():   
    header = st.container()
    model = get_predictor_model()
    st.markdown('<div class="title">Bienvenue dans l\'application pour détecter la violence, Mr {}</div>'.format(name), unsafe_allow_html=True)

    with st.sidebar:

        st.markdown(
        """
        <div class="im1">
            <img src="https://i.ibb.co/kh3LP0F/icon.png" style="width: 150px; height: 150px; margin-right: 10px;">
        </div>
        """,
        unsafe_allow_html=True,
        )
        selected_option = option_menu.option_menu(
            menu_title="",  # Titre du menu
            options=["Tableau de bord", "Traitement d'Image/Vidéo", "Traitement en temps réel", "Historique","ChatBot"],  # Liste des options
            default_index=0,  # Index de l'option par défaut
            orientation="vertical",  # Orientation du menu
            icons=["speedometer2", "camera", "camera-reels","hourglass","robot"],
            styles={
        "container": { 
            "background-color": "#89BEEA",
            "border-radius": "0px",
            "box-shadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2)"
        },                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "27px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",

            },
            "width":"200%",
            "nav-link-selected": { "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee","background-color": "#041c44"},
            },  # Facultatif
        )
    if selected_option == "ChatBot":
        st.title('Chatbot')
        components.iframe("https://seddam.mninou.com/", width=700, height=400)
    elif selected_option   == "Tableau de bord":
        st.title('Tableau de bord')
        st.markdown('<p>Cette option offre une analyse complète des vidéos enregistrées.Elle offre la possibilité de filtrer les vidéos par étiquette, comme ‘Tous’, violence, ou ‘fire’. Les informations sont présentées sous forme de graphiques visuels, y compris un diagramme circulaire montrant la répartition des vidéos de jour et de nuit, et un graphique linéaire montrant le nombre de vidéos ajoutées chaque jour.</p>', unsafe_allow_html=True)

        # Get the list of all date folders in the 'recordings' directory
        date_folders = os.listdir('recordings')

        # Initialize counters for day and night videos
        day_videos = 0
        night_videos = 0
        today_videos = 0  # Initialize a counter for today's videos

        # Add a dropdown menu for label filtering
        labels = ['Tous', 'violence', 'fire']
        selected_label = st.selectbox('Sélectionnez une étiquette', options=labels)
        now = datetime.now()

        today_date = now.strftime("%Y-%m-%d")
        # Iterate over each date folder
        for date_folder in date_folders:
            date_folder_path = os.path.join('recordings', date_folder)

            # Check if the path is a directory
            if os.path.isdir(date_folder_path):
                # Get the list of all video files in the date folder
                video_files = os.listdir(date_folder_path)

                # Iterate over each video file
                for video_file in video_files:
                    # Check if the file is a video
                    if video_file.endswith('.mp4'):
                        # Extract the label and time from the filename
                        label, time_string = video_file.split('_')[:2]

                        # Filter videos by label
                        if label.lower() == selected_label.lower() or selected_label == 'Tous':
                            time_hour = int(time_string.split('-')[0])

                            # Check if the video is day or night
                            if 20 <= time_hour or time_hour < 7:
                                night_videos += 1
                            else:
                                day_videos += 1

                            # Check if the video was added today
                            if date_folder == today_date:
                                today_videos += 1

        # Calculate the percentages
        total_videos = day_videos + night_videos
        day_percentage = (day_videos / total_videos) * 100
        night_percentage = (night_videos / total_videos) * 100
        today_percentage = (today_videos / total_videos) * 100  # Calculate the percentage of videos added today

        # Create a dataframe for the percentages
        df = pd.DataFrame({'Heure': ['Jour', 'Nuit'], '': [day_percentage, night_percentage]})

        # Create the plot
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#Dceaf6')
        df.plot(kind='pie', y='', labels=df['Heure'], autopct='%1.1f%%', ax=ax)

        # Save the plot to a .png file
        fig.savefig("plot.png")

        # Convert the plot to a base64 string
        import base64
        with open("plot.png", "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()

        # Define the CSS for the div
        css = """
        <div style="
            display: block;
            width: 500px;
            height: 500px;
            background-color: #Dceaf6;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            background-image: url('data:image/png;base64,iVBORw0KG...');
        ">
        </div>
        """

        # Replace the placeholder base64 string in the CSS with the actual base64 string of the plot
        css = css.replace("iVBORw0KG...", b64_string)

        # Initialize a dictionary to store the counts
        video_counts = {}

        # Iterate over each date folder
        for date_folder in date_folders:
            date_folder_path = os.path.join('recordings', date_folder)

            # Check if the path is a directory
            if os.path.isdir(date_folder_path):
                # Get the list of all video files in the date folder
                video_files = os.listdir(date_folder_path)

                # Initialize the count for the current date
                video_counts[date_folder] = 0

                # Iterate over each video file
                for video_file in video_files:
                    # Check if the file is a video
                    if video_file.endswith('.mp4'):
                        # Extract the label from the filename
                        label = video_file.split('_')[0]

                        # Filter videos by label
                        if label.lower() == selected_label.lower() or selected_label == 'Tous':
                            # Increment the count for the current date
                            video_counts[date_folder] += 1

        # Convert the dictionary to a pandas DataFrame
        df_count = pd.DataFrame(list(video_counts.items()), columns=['Date', 'Count'])

        # Convert the 'Date' column to datetime
        df_count['Date'] = pd.to_datetime(df_count['Date'])

        # Sort the DataFrame by date
        df_count = df_count.sort_values('Date')

        # Create the plot for line graph
        fig_count, ax_count = plt.subplots()
        fig_count.patch.set_facecolor('#Dceaf6')

        ax_count.plot(df_count['Date'], df_count['Count'])
        plt.xticks(rotation=90)

        # Save the line graph to a .png file
        fig_count.savefig("line_plot.png")

        # Convert the line graph to a base64 string
        with open("line_plot.png", "rb") as img_file:
            b64_string_line = base64.b64encode(img_file.read()).decode()

        # Define the CSS for the div for line graph
        css_line = """
        <div style="
            display: block;
            width: 450px;
            height: 450px;
            background-color: #Dceaf6;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            background-image: url('data:image/png;base64,iVBORw0KG...');
        ">
        </div>
        """

        # Replace the placeholder base64 string in the CSS with the actual base64 string of the line graph
        css_line = css_line.replace("iVBORw0KG...", b64_string_line)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the div in the first column
        col1.markdown(css, unsafe_allow_html=True)

        # Display the div for line graph in the second column
        col2.markdown(css_line, unsafe_allow_html=True)
    elif selected_option == "Traitement d'Image/Vidéo":
        # Uploader de fichiers
        st.title('Traitement d\'Image/Vidéo')
        st.markdown('<p>Cette option permet aux utilisateurs de télécharger des fichiers image ou vidéo pour les traiter. Une fois un fichier téléchargé, l\'application affiche son label prédit. Si le fichier est une vidéo, elle est traitée pour détecter des éléments spécifiques. Si c\'est une image, elle est également analysée pour prédire son label.</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("choisissez une image ou une vidéo...", type=['jpg','png','mp4'])
        if uploaded_file is not None:
            file_details = {"Nom du fichier":uploaded_file.name,"Type de fichier":uploaded_file.type,"Taille du fichier":uploaded_file.size}

            # Traiter le fichier vidéo ou image
            if uploaded_file.type == "video/mp4":
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                process_video(tfile.name)
            else:
                image = Image.open(uploaded_file).convert('RGB')
                image = np.array(image)
                label_text = model.predict(image=image)['label'].title()
                st.write(f'Étiquette prédite : **{label_text}**')
                st.write('Image Originale')
                if len(image.shape) == 3:
                    cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image)

    elif selected_option == "Traitement en temps réel":
        st.title('Traitement en temps réel')
        st.markdown('<p>Cette option permet de démarrer le traitement instantané des données provenant d\'une source en direct, comme une webcam. Une fois activé, l\'application analyse en continu les données en temps réel, permettant une surveillance en direct et une réaction immédiate à tout événement détecté. Les vidéos contenant des événements marqués comme "violence" ou "fire" sont automatiquement enregistrées dans l\'historique pour une revue ultérieure.</p>', unsafe_allow_html=True)
        # Bouton pour démarrer le traitement en temps réel
        if st.button('Démarrer le traitement en temps réel', type="primary"):
            process_realtime()
    elif selected_option == "Historique":
        st.title('Historique')
        date_folders = os.listdir('recordings')
        all_videos = []

        for date_folder in date_folders:
            date_folder_path = os.path.join('recordings', date_folder)
            if os.path.isdir(date_folder_path):
                video_files = os.listdir(date_folder_path)
                for video_file in video_files:
                    if video_file.endswith('.mp4'):
                        video_file_path = os.path.join(date_folder_path, video_file)
                        # Extraire l'étiquette et l'heure du nom de fichier
                        label, time_string = video_file.split('_')[:2]
                        all_videos.append((date_folder, label, time_string, video_file_path))

        selected_videos = st.multiselect('Sélectionnez les vidéos à supprimer', options=all_videos, format_func=lambda x: f'{x[0]} - {x[1]} - {x[2]}')

        if st.button('Supprimer les vidéos sélectionnées', type="primary"):
            for video in selected_videos:
                os.remove(video[3])
                st.success(f'Supprimé {video[0]} - {video[1]} - {video[2]}')
            st.experimental_rerun()

        # Ajouter l'option 'Toutes les vidéos' au menu déroulant
        date_folders.append('Toutes les vidéos')
        selected_title = st.selectbox('Sélectionnez une date', options=date_folders, index=len(date_folders)-1)

        # Ajouter un menu déroulant pour le filtrage par étiquette
        labels = ['Tous', 'violence', 'fire']
        selected_label = st.selectbox('Sélectionnez une étiquette', options=labels)

        for date_folder in date_folders:
            date_folder_path = os.path.join('recordings', date_folder)
            if os.path.isdir(date_folder_path) and (date_folder == selected_title or selected_title == 'Toutes les vidéos'):
                with st.container():
                    st.title(date_folder)
                    video_files = os.listdir(date_folder_path)
                    for video_file in video_files:
                        if video_file.endswith('.mp4'):
                            video_file_path = os.path.join(date_folder_path, video_file)
                            # Extraire l'étiquette et l'heure du nom de fichier
                            label, time_string = video_file.split('_')[:2]
                            # Filtrer les vidéos par étiquette
                            if label.lower() == selected_label.lower() or selected_label == 'Tous':
                                with st.expander(f"{label} - {time_string}"):
                                    with open(video_file_path, "rb") as f:
                                        video_bytes = f.read()
                                    st.video(video_bytes)

# Charger le fichier YAML
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Créer l'objet d'authentification
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
login_page_css()

with st.container():
    name, authentication_status, username = authenticator.login(fields='main')
    # Authentifier les utilisateurs
    if authentication_status:
        main()
        authenticator.logout('Logout', 'sidebar')
    elif authentication_status == False:
        st.markdown('<div class="login_error">Nom d\'utilisateur/mot de passe incorrect</div>', unsafe_allow_html=True)
    elif authentication_status == None:
        st.markdown('<div class="login_warning">Veuillez entrer votre nom d\'utilisateur et votre mot de passe</div>', unsafe_allow_html=True)
