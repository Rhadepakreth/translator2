#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test direct pour l'intégration Mistral LLM
Teste l'analyse de prompt et la traduction sans interface audio
"""

import os
import sys
from dotenv import load_dotenv
from mistralai import Mistral
from deep_translator import GoogleTranslator
import json

# Charger les variables d'environnement
load_dotenv()

def test_mistral_analysis(prompt_text):
    """
    Teste l'analyse d'un prompt par Mistral AI
    """
    print(f"\n🔍 Test d'analyse du prompt: '{prompt_text}'")
    print("=" * 60)
    
    # Initialiser le client Mistral
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("❌ Erreur: Clé API Mistral non configurée")
        print("💡 Ajoutez MISTRAL_API_KEY=votre_clé dans le fichier .env")
        return None
    
    try:
        client = Mistral(api_key=api_key)
        print("✅ Client Mistral initialisé avec succès")
        
        # Prompt système pour l'analyse
        system_prompt = """
Tu es un assistant spécialisé dans l'analyse de demandes de traduction.
Analyse le texte fourni et extrais :
1. La tâche demandée (traduction, explication, etc.)
2. La langue source (si mentionnée ou détectable)
3. La langue cible (si mentionnée)
4. Le texte exact à traiter

Réponds UNIQUEMENT en JSON avec cette structure :
{
  "task": "traduction|explication|autre",
  "source_language": "langue_source_ou_auto",
  "target_language": "langue_cible_ou_null",
  "text_to_process": "texte_exact_à_traiter",
  "confidence": 0.95
}
"""
        
        # Envoyer la requête à Mistral
        print("📤 Envoi de la requête à Mistral...")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        # Extraire et parser la réponse
        raw_response = response.choices[0].message.content.strip()
        print(f"📥 Réponse brute de Mistral:\n{raw_response}")
        
        # Parser le JSON
        try:
            analysis = json.loads(raw_response)
            print("\n✅ Analyse JSON parsée avec succès:")
            print(f"   📋 Tâche: {analysis.get('task', 'Non détectée')}")
            print(f"   🔤 Langue source: {analysis.get('source_language', 'Non détectée')}")
            print(f"   🎯 Langue cible: {analysis.get('target_language', 'Non détectée')}")
            print(f"   📝 Texte à traiter: '{analysis.get('text_to_process', 'Non détecté')}'")
            print(f"   🎯 Confiance: {analysis.get('confidence', 'Non spécifiée')}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Erreur de parsing JSON: {e}")
            print("🔧 Tentative de nettoyage de la réponse...")
            
            # Tentative de nettoyage
            cleaned = raw_response
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1].split('```')[0]
            
            try:
                analysis = json.loads(cleaned.strip())
                print("✅ Parsing réussi après nettoyage")
                return analysis
            except:
                print("❌ Impossible de parser la réponse même après nettoyage")
                return None
                
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à Mistral: {e}")
        return None

def test_translation(text, target_lang):
    """
    Teste la traduction avec Google Translate
    """
    print(f"\n🌐 Test de traduction vers {target_lang}")
    print("=" * 40)
    
    try:
        # Mapper les langues vers les codes ISO
        lang_mapping = {
            'français': 'fr', 'french': 'fr',
            'anglais': 'en', 'english': 'en',
            'espagnol': 'es', 'spanish': 'es',
            'allemand': 'de', 'german': 'de',
            'italien': 'it', 'italian': 'it',
            'suédois': 'sv', 'swedish': 'sv',
            'norvégien': 'no', 'norwegian': 'no',
            'danois': 'da', 'danish': 'da'
        }
        
        target_code = lang_mapping.get(target_lang.lower(), target_lang.lower())
        print(f"📝 Texte à traduire: '{text}'")
        print(f"🎯 Code langue cible: {target_code}")
        
        translator = GoogleTranslator(source='auto', target=target_code)
        translation = translator.translate(text)
        
        print(f"✅ Traduction réussie: '{translation}'")
        return translation
        
    except Exception as e:
        print(f"❌ Erreur de traduction: {e}")
        return None

def run_complete_test(prompt):
    """
    Exécute un test complet: analyse + traduction
    """
    print(f"\n🚀 TEST COMPLET")
    print("=" * 80)
    
    # Étape 1: Analyse du prompt
    analysis = test_mistral_analysis(prompt)
    if not analysis:
        print("❌ Test échoué à l'étape d'analyse")
        return False
    
    # Étape 2: Traduction si demandée
    if analysis.get('task') == 'traduction' and analysis.get('text_to_process') and analysis.get('target_language'):
        translation = test_translation(
            analysis['text_to_process'], 
            analysis['target_language']
        )
        
        if translation:
            print(f"\n🎉 RÉSULTAT FINAL:")
            print(f"   📝 Texte original: '{analysis['text_to_process']}'")
            print(f"   🌐 Traduction en {analysis['target_language']}: '{translation}'")
            return True
        else:
            print("❌ Test échoué à l'étape de traduction")
            return False
    else:
        print("ℹ️  Pas de traduction demandée ou informations insuffisantes")
        return True

if __name__ == "__main__":
    # Tests prédéfinis
    test_prompts = [
        "Comment dit-on j'aime les brocolis sans sel en suédois",
        "Traduis 'Hello world' en français",
        "Peux-tu me dire comment on dit 'bonne nuit' en espagnol",
        "What is 'merci beaucoup' in English"
    ]
    
    print("🧪 TESTS DIRECTS DE L'INTÉGRATION MISTRAL LLM")
    print("=" * 80)
    
    # Test avec prompt personnalisé si fourni en argument
    if len(sys.argv) > 1:
        custom_prompt = " ".join(sys.argv[1:])
        print(f"📝 Test avec prompt personnalisé: '{custom_prompt}'")
        run_complete_test(custom_prompt)
    else:
        # Tests avec prompts prédéfinis
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📋 TEST {i}/{len(test_prompts)}")
            success = run_complete_test(prompt)
            
            if not success:
                print(f"❌ Test {i} échoué")
            else:
                print(f"✅ Test {i} réussi")
            
            print("\n" + "-" * 80)
    
    print("\n🏁 Tests terminés")