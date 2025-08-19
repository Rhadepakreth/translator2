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
from deep_translator import GoogleTranslator
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
    
    # Modèle de traduction français -> anglais (gardé pour compatibilité)
    print("Chargement du modèle de traduction...")
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name)
    translation_model.to(device)
    
    # Reconnaissance vocale
    speech_recognizer = sr.Recognizer()
    
    print("Tous les modèles sont initialisés!")

def is_text_corrupted(text):
    """Vérifie si le texte traduit est corrompu (répétitions anormales, caractères étranges)"""
    if not text or len(text.strip()) == 0:
        return True
    
    # Détecter les répétitions excessives de mots
    words = text.split()
    if len(words) > 3:
        # Compter les mots répétés consécutivement
        consecutive_repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_repeats += 1
                if consecutive_repeats >= 2:  # Plus de 2 répétitions consécutives
                    return True
            else:
                consecutive_repeats = 0
    
    # Détecter les caractères de contrôle ou encodage bizarre
    if any(ord(char) < 32 and char not in '\n\r\t' for char in text):
        return True
    
    # Détecter les répétitions de syllabes (comme "Tagalag Tagalag")
    # Exclure les langues asiatiques qui peuvent avoir des structures différentes
    has_asian_chars = any(ord(char) > 0x4E00 for char in text)  # Caractères CJK
    if not has_asian_chars and len(set(words)) < len(words) * 0.5 and len(words) > 4:
        return True
    
    return False

def translate_text(text, target_lang='en'):
    """Traduit le texte français vers la langue cible spécifiée avec validation"""
    
    try:
        # Mapping pour corriger les codes de langue chinois
        lang_mapping = {
            'zh-cn': 'zh-CN',  # Chinois simplifié
            'zh-tw': 'zh-TW'   # Chinois traditionnel
        }
        
        # Utiliser le mapping si nécessaire
        mapped_lang = lang_mapping.get(target_lang, target_lang)
        
        # Utilisation de GoogleTranslator de deep_translator pour supporter plus de langues
        translator = GoogleTranslator(source='fr', target=mapped_lang)
        result = translator.translate(text)
        
        # Validation du résultat
        if is_text_corrupted(result):
            print(f"Texte traduit corrompu détecté pour {target_lang}: {result}")
            # Essayer une traduction en anglais comme fallback
            if target_lang != 'en':
                print(f"Fallback vers l'anglais pour éviter le texte corrompu")
                en_translator = GoogleTranslator(source='fr', target='en')
                fallback_result = en_translator.translate(text)
                if not is_text_corrupted(fallback_result):
                    return fallback_result
            return "Traduction non disponible pour cette langue"
        
        return result
    
    except Exception as e:
        print(f"Erreur lors de la traduction vers {target_lang}: {e}")
        # Fallback vers le modèle Marian pour l'anglais si Google Translate échoue
        if target_lang == 'en' and translation_model and translation_tokenizer:
            try:
                inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                # Déplacer vers le bon device
                device = next(translation_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = translation_model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated_text
            except Exception as fallback_error:
                print(f"Erreur du fallback Marian: {fallback_error}")
        
        return "Erreur de traduction"

def synthesize_speech(text, lang='en'):
    """Synthétise la parole à partir du texte traduit dans la langue spécifiée"""
    try:
        print(f"Synthèse vocale demandée pour: {text}")
        
        # Vérifier si le texte est valide avant la synthèse
        if is_text_corrupted(text):
            print(f"Texte corrompu détecté, impossible de synthétiser: {text}")
            raise ValueError("Texte corrompu détecté")
        
        # Mapping des codes de langue pour gTTS (certaines langues utilisent des codes différents)
        lang_mapping = {
            'zh-cn': 'zh',  # Chinois simplifié
            'zh-tw': 'zh-tw',  # Chinois traditionnel
            'pt-br': 'pt',  # Portugais brésilien -> portugais
            'nb': 'no',  # Norvégien bokmål -> norvégien
            'nn': 'no',  # Norvégien nynorsk -> norvégien
        }
        
        # Langues supportées par gTTS (liste non exhaustive des principales)
        supported_langs = {
            'af', 'ar', 'bg', 'bn', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
            'eo', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'hr', 'hu', 'hy', 'id', 'is', 
            'it', 'ja', 'jw', 'km', 'kn', 'ko', 'la', 'lv', 'mk', 'ml', 'mr', 'my', 
            'ne', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'si', 'sk', 'sq', 'sr', 'su', 
            'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh', 'zh-tw'
        }
        
        # Utiliser le mapping si disponible, sinon utiliser le code original
        gtts_lang = lang_mapping.get(lang, lang)
        
        # Vérifier si la langue est supportée par gTTS
        if gtts_lang not in supported_langs:
            print(f"Langue {gtts_lang} non supportée par gTTS, fallback vers l'anglais")
            gtts_lang = 'en'
        
        # Créer un objet gTTS pour la langue spécifiée
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Sauvegarde dans un buffer en mémoire
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encodage en base64 pour l'envoi au frontend
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        return audio_base64
            
    except Exception as e:
        print(f"Erreur lors de la synthèse vocale en {lang}: {e}")
        # Fallback vers l'anglais si la langue n'est pas supportée
        try:
            print("Tentative de fallback vers l'anglais...")
            fallback_tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = io.BytesIO()
            fallback_tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
            return audio_base64
        except Exception as fallback_error:
            print(f"Erreur du fallback anglais: {fallback_error}")
            raise Exception(f"Erreur lors de la génération audio: {str(e)}")
        
        raise Exception(f"Erreur lors de la génération audio: {str(e)}")

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
        target_lang = data.get('target_lang', 'en')  # Langue cible, anglais par défaut
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Traduction
        translated_text = translate_text(text, target_lang)
        
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
        lang = data.get('lang', 'en')  # Langue pour la synthèse vocale
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Synthèse vocale avec gTTS
        try:
            audio_base64 = synthesize_speech(text, lang)
            return jsonify({
                'audio': audio_base64,
                'success': True
            })
        except Exception as synthesis_error:
            print(f"Erreur de synthèse vocale: {synthesis_error}")
            return jsonify({
                'error': f'Erreur lors de la génération audio: {str(synthesis_error)}',
                'text': text,
                'success': False
            }), 500
    
    except Exception as e:
        print(f"Erreur générale dans /synthesize: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/translate_and_synthesize', methods=['POST'])
def translate_and_synthesize():
    """Route pour traduction + synthèse vocale en une seule étape"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        target_lang = data.get('target_lang', 'en')  # Langue cible pour la traduction
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Traduction
        translated_text = translate_text(text, target_lang)
        
        # Synthèse vocale dans la langue cible
        audio_base64 = synthesize_speech(translated_text, target_lang)
        
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
    """Route pour traiter l'audio (reconnaissance + traduction + synthèse)"""
    try:
        data = request.get_json()
        audio_data = data.get('audio')
        target_lang = data.get('target_lang', 'en')  # Langue cible pour la traduction
        
        if not audio_data:
            return jsonify({'error': 'Aucune donnée audio fournie'}), 400
        
        # Décoder les données audio base64
        try:
            audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        except Exception as e:
            return jsonify({'error': f'Erreur de décodage audio: {str(e)}'}), 400
        
        # Sauvegarder temporairement le fichier audio
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Reconnaissance vocale
            recognized_text = recognize_speech_from_file(temp_filename)
            
            if not recognized_text or recognized_text == "Erreur de reconnaissance vocale":
                return jsonify({'error': 'Impossible de reconnaître la parole'}), 400
            
            # Traduction vers la langue cible
            translated_text = translate_text(recognized_text, target_lang)
            
            # Synthèse vocale dans la langue cible
            audio_base64 = synthesize_speech(translated_text, target_lang)
            
            return jsonify({
                'recognized_text': recognized_text,
                'translated_text': translated_text,
                'audio': audio_base64
            })
        
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    except Exception as e:
        print(f"Erreur dans /process_audio: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialisation des modèles au démarrage
    init_models()
    
    # Lancement de l'application
    app.run(debug=True, host='0.0.0.0', port=8080)