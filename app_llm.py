import os
import io
import json
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
from mistralai import Mistral
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Chargement des variables d'environnement
load_dotenv()

# Configuration pour éviter les erreurs de compilation
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
speech_recognizer = None
mistral_client = None

def init_models():
    """Initialise tous les modèles nécessaires"""
    global translation_model, translation_tokenizer, speech_recognizer, mistral_client
    
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
    
    # Reconnaissance vocale avec paramètres optimisés
    speech_recognizer = sr.Recognizer()
    
    # Configuration optimale pour la reconnaissance vocale
    speech_recognizer.energy_threshold = 300  # Seuil d'énergie pour détecter la parole
    speech_recognizer.dynamic_energy_threshold = True  # Ajustement automatique du seuil
    speech_recognizer.pause_threshold = 0.8  # Durée de pause pour considérer la fin d'une phrase
    speech_recognizer.phrase_threshold = 0.3  # Durée minimale pour considérer un son comme parole
    speech_recognizer.non_speaking_duration = 0.5  # Durée de silence avant d'arrêter l'écoute
    
    # Initialisation du client Mistral
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key or api_key == 'your_mistral_api_key_here':
        print("⚠️  ATTENTION: Clé API Mistral non configurée dans .env")
        print("   Veuillez configurer MISTRAL_API_KEY dans le fichier .env")
        print("🔄 Mode fallback activé: analyse basique sans IA")
        mistral_client = None
    else:
        try:
            mistral_client = Mistral(api_key=api_key)
            print("✅ Client Mistral initialisé avec succès")
            print("🤖 Mode IA activé: analyse intelligente des prompts")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de Mistral: {e}")
            print("🔄 Mode fallback activé: analyse basique sans IA")
            mistral_client = None
    
    print("Tous les modèles sont initialisés avec des paramètres optimisés!")

def is_text_corrupted(text):
    """Vérifie si le texte contient des caractères corrompus ou incohérents"""
    if not text or len(text.strip()) == 0:
        return True
    
    # Vérifier les caractères suspects
    suspicious_chars = ['�', '\ufffd', '\x00']
    for char in suspicious_chars:
        if char in text:
            return True
    
    # Vérifier si le texte est principalement composé de caractères non-alphabétiques
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) > 10 and alpha_count / len(text) < 0.3:
        return True
    
    return False

def analyze_prompt_with_mistral(text):
    """Analyse le prompt utilisateur avec Mistral pour extraire la tâche, langues et texte à traduire"""
    if not mistral_client:
        # Fallback si Mistral n'est pas disponible
        print("🔄 Utilisation du mode fallback (Mistral non disponible)")
        print(f"📝 Texte analysé en mode basique: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        return {
            'task': 'translate',
            'source_language': 'auto',
            'target_language': 'en',
            'text_to_translate': text,
            'confidence': 0.5,
            'method': 'fallback'
        }
    
    try:
        # Prompt système pour Mistral
        system_prompt = """Tu es un assistant spécialisé dans l'analyse de demandes de traduction.
Ton rôle est d'analyser un prompt utilisateur et d'extraire :
- La tâche demandée (translate, explain, etc.)
- La langue source (code ISO 639-1 ou 'auto' si non spécifiée)
- La langue cible (code ISO 639-1)
- Le texte exact à traduire

Réponds UNIQUEMENT avec un JSON valide au format :
{
  "task": "translate",
  "source_language": "fr",
  "target_language": "en", 
  "text_to_translate": "texte exact",
  "confidence": 0.95
}

Exemples :
- "traduis bonjour en anglais" → {"task":"translate","source_language":"fr","target_language":"en","text_to_translate":"bonjour","confidence":0.9}
- "comment dit-on merci en japonais" → {"task":"translate","source_language":"fr","target_language":"ja","text_to_translate":"merci","confidence":0.85}
- "translate hello to spanish" → {"task":"translate","source_language":"en","target_language":"es","text_to_translate":"hello","confidence":0.9}"""
        
        # Appel à l'API Mistral
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyse ce prompt: '{text}'"}
        ]
        
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=0.1,
            max_tokens=200
        )
        
        # Extraction de la réponse
        response_text = response.choices[0].message.content.strip()
        
        # Nettoyage de la réponse (enlever les balises markdown si présentes)
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        result = json.loads(response_text)
        result['method'] = 'mistral'
        
        print("🤖 Analyse effectuée par Mistral IA")
        print(f"📝 Texte analysé: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"✅ Résultat Mistral: {result}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur parsing JSON Mistral: {e}")
        print(f"📄 Réponse brute: {response_text}")
        print("🔄 Basculement vers le mode fallback")
    except Exception as e:
        print(f"❌ Erreur Mistral API: {e}")
        print("🔄 Basculement vers le mode fallback")
    
    # Fallback en cas d'erreur
    print(f"📝 Texte analysé en mode fallback (erreur): '{text[:50]}{'...' if len(text) > 50 else ''}'")
    return {
        'task': 'translate',
        'source_language': 'auto',
        'target_language': 'en',
        'text_to_translate': text,
        'confidence': 0.3,
        'method': 'fallback_error'
    }

def translate_text(text, target_lang='en', source_lang='auto'):
    """Traduit le texte vers la langue cible spécifiée"""
    try:
        # Utilisation de GoogleTranslator pour plus de langues
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        return {
            'success': True,
            'translated_text': translated,
            'source_language': source_lang,
            'target_language': target_lang
        }
    except Exception as e:
        print(f"Erreur de traduction: {e}")
        return {
            'success': False,
            'error': str(e),
            'original_text': text
        }

def synthesize_speech(text, lang='en'):
    """Synthétise la parole à partir du texte"""
    try:
        # Mapping des codes de langue pour gTTS
        lang_mapping = {
            'en': 'en',
            'fr': 'fr', 
            'es': 'es',
            'de': 'de',
            'it': 'it',
            'ja': 'ja',
            'zh': 'zh',
            'pt': 'pt',
            'ru': 'ru',
            'ar': 'ar'
        }
        
        gtts_lang = lang_mapping.get(lang, 'en')
        
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            
            # Lecture du fichier audio
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Suppression du fichier temporaire
            os.unlink(tmp_file.name)
            
            return {
                'success': True,
                'audio_data': base64.b64encode(audio_data).decode('utf-8'),
                'language': gtts_lang
            }
            
    except Exception as e:
        print(f"Erreur de synthèse vocale: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def recognize_speech_from_file(audio_file_path):
    """Reconnaissance vocale améliorée avec preprocessing audio optimisé"""
    try:
        print(f"🎤 Traitement du fichier audio: {audio_file_path}")
        
        # Chargement et preprocessing de l'audio avec pydub
        audio = AudioSegment.from_file(audio_file_path)
        
        # Normalisation du volume (ajustement automatique)
        normalized_audio = audio.normalize()
        
        # Augmentation du volume si nécessaire
        if normalized_audio.dBFS < -20:
            normalized_audio = normalized_audio + (abs(normalized_audio.dBFS + 14))
            print(f"📈 Volume augmenté de {abs(normalized_audio.dBFS + 14):.1f}dB")
        
        # Conversion en WAV optimisé pour la reconnaissance
        wav_audio = normalized_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Sauvegarde temporaire du fichier WAV optimisé
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_audio.export(temp_wav.name, format="wav")
            temp_wav_path = temp_wav.name
        
        try:
            # Reconnaissance vocale avec ajustement du bruit ambiant
            with sr.AudioFile(temp_wav_path) as source:
                # Ajustement automatique du bruit ambiant
                speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print(f"🔧 Seuil d'énergie ajusté: {speech_recognizer.energy_threshold}")
                
                # Enregistrement de l'audio
                audio_data = speech_recognizer.record(source)
            
            # Tentative de reconnaissance avec plusieurs langues
            languages = ['fr-FR', 'en-US', 'es-ES', 'it-IT', 'de-DE']
            
            for lang in languages:
                try:
                    print(f"🌍 Tentative de reconnaissance en {lang}...")
                    result = speech_recognizer.recognize_google(
                        audio_data, 
                        language=lang,
                        show_all=False
                    )
                    
                    if result and not is_text_corrupted(result):
                        print(f"✅ Reconnaissance réussie en {lang}: '{result}'")
                        return {
                            'success': True,
                            'text': result,
                            'language': lang,
                            'confidence': 'high'
                        }
                        
                except sr.UnknownValueError:
                    print(f"❌ Pas de reconnaissance en {lang}")
                    continue
                except sr.RequestError as e:
                    print(f"❌ Erreur de requête pour {lang}: {e}")
                    continue
            
            # Si aucune langue n'a fonctionné
            return {
                'success': False,
                'error': 'Impossible de comprendre l\'audio dans les langues supportées',
                'details': 'Aucune reconnaissance réussie'
            }
            
        finally:
            # Nettoyage du fichier temporaire
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
                
    except Exception as e:
        print(f"❌ Erreur lors de la reconnaissance vocale: {e}")
        return {
            'success': False,
            'error': f'Erreur de traitement audio: {str(e)}'
        }

# Routes Flask
@app.route('/')
def index():
    return render_template('index_llm.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Route pour la traduction simple (sans LLM)"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')
        source_lang = data.get('source_lang', 'auto')
        
        if not text:
            return jsonify({'success': False, 'error': 'Texte manquant'})
        
        result = translate_text(text, target_lang, source_lang)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Route pour la synthèse vocale"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        lang = data.get('lang', 'en')
        
        if not text:
            return jsonify({'success': False, 'error': 'Texte manquant'})
        
        result = synthesize_speech(text, lang)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process_audio_llm', methods=['POST'])
def process_audio_llm():
    """Route principale pour le traitement audio avec analyse LLM"""
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        
        if not audio_data:
            return jsonify({'success': False, 'error': 'Données audio manquantes'})
        
        # Décodage des données audio base64
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Erreur de décodage audio: {str(e)}'})
        
        # Sauvegarde temporaire du fichier audio
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        
        try:
            # Étape 1: Reconnaissance vocale
            print("🎤 Étape 1: Reconnaissance vocale...")
            speech_result = recognize_speech_from_file(temp_audio_path)
            
            if not speech_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'Reconnaissance vocale échouée',
                    'details': speech_result.get('error', 'Erreur inconnue')
                })
            
            recognized_text = speech_result['text']
            print(f"✅ Texte reconnu: '{recognized_text}'")
            
            # Étape 2: Analyse du prompt avec Mistral
            print("🧠 Étape 2: Analyse du prompt avec Mistral...")
            analysis_result = analyze_prompt_with_mistral(recognized_text)
            print(f"✅ Analyse terminée: {analysis_result}")
            
            # Étape 3: Traduction du texte extrait
            print("🌍 Étape 3: Traduction...")
            translation_result = translate_text(
                analysis_result['text_to_translate'],
                analysis_result['target_language'],
                analysis_result['source_language']
            )
            
            if not translation_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'Traduction échouée',
                    'details': translation_result.get('error', 'Erreur inconnue')
                })
            
            translated_text = translation_result['translated_text']
            print(f"✅ Traduction: '{translated_text}'")
            
            # Étape 4: Synthèse vocale
            print("🔊 Étape 4: Synthèse vocale...")
            synthesis_result = synthesize_speech(translated_text, analysis_result['target_language'])
            
            if not synthesis_result['success']:
                return jsonify({
                    'success': False,
                    'error': 'Synthèse vocale échouée',
                    'details': synthesis_result.get('error', 'Erreur inconnue')
                })
            
            # Résultat final complet
            return jsonify({
                'success': True,
                'original_text': recognized_text,
                'analysis': analysis_result,
                'translated_text': translated_text,
                'audio_data': synthesis_result['audio_data'],
                'processing_steps': {
                    'speech_recognition': speech_result,
                    'llm_analysis': analysis_result,
                    'translation': translation_result,
                    'synthesis': synthesis_result
                }
            })
            
        finally:
            # Nettoyage du fichier temporaire
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        print(f"❌ Erreur dans process_audio_llm: {e}")
        return jsonify({
            'success': False,
            'error': f'Erreur de traitement: {str(e)}'
        })

if __name__ == '__main__':
    print("🚀 Démarrage de l'application avec intégration LLM Mistral...")
    init_models()
    print("🌐 Serveur disponible sur: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)