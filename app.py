import os
import io
import base64
import tempfile
import torch
import torchaudio
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify, send_file
from transformers import MarianMTModel, MarianTokenizer
from werkzeug.utils import secure_filename
from gtts import gTTS
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

# Configuration pour Mac M2 sans CUDA
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales pour les modèles
translation_model = None
translation_tokenizer = None
csm_generator = None
speech_recognizer = None
prompt_segment = None

def init_models():
    """Initialise tous les modèles nécessaires"""
    global translation_model, translation_tokenizer, csm_generator, speech_recognizer, prompt_segment
    
    print("Initialisation des modèles...")
    
    # Détection du device optimal pour Mac M2
    if torch.backends.mps.is_available():
        device = "mps"
        print("Utilisation de MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        print("Utilisation du CPU")
    
    # Modèle de traduction français -> anglais
    print("Chargement du modèle de traduction...")
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    translation_model.to(device)
    
    # Reconnaissance vocale
    speech_recognizer = sr.Recognizer()
    
    print("Tous les modèles sont initialisés!")

def translate_text(text, source_lang="fr", target_lang="en"):
    """Traduit le texte du français vers l'anglais"""
    try:
        # Préparation du texte pour la traduction
        inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Déplacer vers le bon device
        device = next(translation_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Génération de la traduction
        with torch.no_grad():
            outputs = translation_model.generate(**inputs, max_length=512, num_beams=4)
        
        # Décodage du résultat
        translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    except Exception as e:
        print(f"Erreur lors de la traduction: {e}")
        return f"Erreur de traduction: {str(e)}"

def synthesize_speech(text):
    """Synthèse vocale avec gTTS"""
    try:
        print(f"Synthèse vocale demandée pour: {text}")
        
        # Créer un objet gTTS pour l'anglais
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tts.save(tmp_file.name)
            
            # Lire le fichier et l'encoder en base64
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Supprimer le fichier temporaire
            os.unlink(tmp_file.name)
            
            return audio_base64
            
    except Exception as e:
        print(f"Erreur lors de la synthèse vocale: {e}")
        return None

def recognize_speech_from_file(audio_file_path):
    """Reconnaissance vocale à partir d'un fichier audio avec conversion automatique"""
    try:
        # Détecter l'extension du fichier
        file_extension = os.path.splitext(audio_file_path)[1].lower()
        
        # Si le fichier n'est pas dans un format supporté par speech_recognition, le convertir
        if file_extension not in ['.wav', '.aiff', '.aif', '.flac']:
            print(f"Conversion du fichier {file_extension} vers WAV...")
            
            # Charger le fichier audio avec pydub
            audio_segment = AudioSegment.from_file(audio_file_path)
            
            # Créer un fichier temporaire WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                wav_path = tmp_wav.name
                # Exporter en WAV
                audio_segment.export(wav_path, format='wav')
            
            # Utiliser le fichier WAV converti
            converted_file_path = wav_path
            should_cleanup = True
        else:
            # Utiliser le fichier original
            converted_file_path = audio_file_path
            should_cleanup = False
        
        # Reconnaissance vocale
        with sr.AudioFile(converted_file_path) as source:
            audio = speech_recognizer.record(source)
            # Reconnaissance en français
            text = speech_recognizer.recognize_google(audio, language='fr-FR')
        
        # Nettoyer le fichier temporaire si nécessaire
        if should_cleanup:
            os.unlink(converted_file_path)
        
        return text
        
    except sr.UnknownValueError:
        return "Impossible de comprendre l'audio"
    except sr.RequestError as e:
        return f"Erreur du service de reconnaissance vocale: {e}"
    except Exception as e:
        return f"Erreur lors de la reconnaissance vocale: {e}"

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Route pour la traduction de texte"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Traduction
        translated_text = translate_text(text)
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Route pour la synthèse vocale"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Synthèse vocale avec gTTS
        audio_base64 = synthesize_speech(text)
        
        if audio_base64 is None:
            return jsonify({
                'error': 'Erreur lors de la génération audio',
                'text': text,
                'success': False
            }), 500
        
        return jsonify({
            'audio_data': audio_base64,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/translate_and_synthesize', methods=['POST'])
def translate_and_synthesize():
    """Route pour traduction + synthèse vocale en une seule étape"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Traduction
        translated_text = translate_text(text)
        
        # Synthèse vocale
        audio_base64 = synthesize_speech(translated_text)
        
        if audio_base64 is None:
            return jsonify({
                'original_text': text,
                'translated_text': translated_text,
                'message': 'Traduction réussie - Synthèse vocale temporairement indisponible',
                'success': True
            })
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'audio_data': audio_base64,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Route pour l'upload et la reconnaissance vocale"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Aucun fichier audio fourni'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Sauvegarde temporaire du fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Reconnaissance vocale
        recognized_text = recognize_speech_from_file(filepath)
        
        # Suppression du fichier temporaire
        os.unlink(filepath)
        
        return jsonify({
            'recognized_text': recognized_text,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Route complète: reconnaissance vocale + traduction depuis enregistrement microphone"""
    try:
        # Vérifier si c'est un fichier uploadé (ancien système) ou des données blob (nouveau système)
        if 'audio' in request.files:
            # Ancien système de fichier uploadé
            file = request.files['audio']
            if file.filename == '':
                return jsonify({'error': 'Aucun fichier sélectionné'}), 400
            
            # Sauvegarde temporaire du fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Reconnaissance vocale
            recognized_text = recognize_speech_from_file(filepath)
            
            # Suppression du fichier temporaire
            os.unlink(filepath)
        else:
            # Nouveau système d'enregistrement microphone
            audio_data = request.get_data()
            if not audio_data:
                return jsonify({'error': 'Aucune donnée audio fournie'}), 400
            
            # Sauvegarde temporaire des données audio
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_filepath = tmp_file.name
            
            # Reconnaissance vocale
            recognized_text = recognize_speech_from_file(tmp_filepath)
            
            # Suppression du fichier temporaire
            os.unlink(tmp_filepath)
        
        if "Erreur" in recognized_text or "Impossible" in recognized_text:
            return jsonify({
                'error': recognized_text,
                'success': False
            }), 400
        
        # Traduction
        translated_text = translate_text(recognized_text)
        
        # Synthèse vocale
        audio_base64 = synthesize_speech(translated_text)
        
        if audio_base64 is None:
            return jsonify({
                'recognized_text': recognized_text,
                'translated_text': translated_text,
                'error': 'Erreur lors de la synthèse vocale',
                'success': False
            }), 500
        
        return jsonify({
            'recognized_text': recognized_text,
            'translated_text': translated_text,
            'audio_data': audio_base64,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # Initialisation des modèles au démarrage
    init_models()
    
    # Lancement de l'application
    app.run(debug=True, host='0.0.0.0', port=8080)