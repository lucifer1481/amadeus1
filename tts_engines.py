# tts_engines.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Text-to-Speech Engines
#   Description: Contains all voice synthesis classes for Amadeus.
# =================================================================================================

import os
import shlex
import subprocess
import re
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
import pyttsx3
import simpleaudio as sa
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
from TTS.api import TTS

from config import Config

class BaseTTS:
    """Base class for all TTS engines to ensure a consistent interface."""
    def __init__(self):
        self.ready = False
    
    def speak(self, text: str):
        raise NotImplementedError("Subclasses must implement the 'speak' method.")

class V1_Core_TTS(BaseTTS):
    """V1 Core: A fast, offline, system-native voice using pyttsx3."""
    def __init__(self):
        super().__init__()
        print("✅ V1 Core Initialized (pyttsx3).")
        self.ready = True

    def speak(self, text: str):
        print(f"Amadeus: {text}")
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error: V1 Core]: {e}")

class V1_5_Core_TTS(BaseTTS):
    """V1.5 Core: A high-quality, natural-sounding cloud voice using ElevenLabs."""
    def __init__(self):
        super().__init__()
        print("Loading V1.5 Core (ElevenLabs)...")
        if not Config.ELEVENLABS_API_KEY.startswith("sk_") or not Config.ELEVENLABS_VOICE_ID:
            print("!!! CRITICAL ERROR: ElevenLabs API Key or Voice ID is not set in config.py.")
            return
        try:
            self.client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
            print("✅ ElevenLabs client initialized.")
            self.ready = True
        except Exception as e:
            print(f"Could not initialize ElevenLabs client. Error: {e}")

    def speak(self, text: str):
        print(f"Amadeus: {text}")
        if not self.ready:
            return
        try:
            audio = self.client.text_to_speech.convert(voice_id=Config.ELEVENLABS_VOICE_ID, text=text)
            play(audio)
        except Exception as e:
            print(f"[TTS Error: V1.5 Core]: {e}")

class V8_Core_TTS(BaseTTS):
    """V8 Genesis Core: Advanced voice cloning using SpeechT5."""
    def __init__(self):
        super().__init__()
        print("Loading V8 Genesis Engine (SpeechT5)...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            spk_model_source = "speechbrain/spkrec-xvect-voxceleb"
            self.spk_model = EncoderClassifier.from_hparams(source=spk_model_source, savedir=os.path.join("models", spk_model_source), run_opts={"device": self.device})
            self.speaker_embeddings = None
            print("✅ Genesis Engine components loaded.")
            self.ready = True
        except Exception as e:
            print(f"Could not initialize Genesis Engine. Error: {e}")

    def create_speaker_embedding(self, wav_file):
        """Creates a voice embedding from a WAV file."""
        audio, _ = sf.read(wav_file)
        with torch.no_grad():
            embedding = self.spk_model.encode_batch(torch.tensor(audio).unsqueeze(0).to(self.device))
            embedding = torch.nn.functional.normalize(embedding, dim=2)
        return embedding.squeeze().cpu().unsqueeze(0)

    def speak(self, text: str):
        print(f"Amadeus: {text}")
        if not self.ready or self.speaker_embeddings is None:
            print("[TTS Warning: Speaker embedding not set for V8 Core.]")
            return
        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings.to(self.device), vocoder=self.vocoder)
            amplified_speech = np.clip(speech.cpu().numpy() * Config.VOLUME_BOOST, -1.0, 1.0)
            sd.play(amplified_speech, samplerate=16000)
            sd.wait()
        except Exception as e:
            print(f"[TTS Error: V8 Core]: {e}")

class V9_Core_TTS(BaseTTS):
    """V9 Rapida Core: A high-quality, fast, and fully offline voice using Piper."""
    def __init__(self):
        super().__init__()
        print("Loading V9 Rapida Engine (Piper TTS)...")
        if not os.path.exists(Config.PIPER_EXE_PATH) or not os.path.exists(Config.PIPER_MODEL_PATH):
            print("!!! CRITICAL ERROR: Piper components not found. Check paths in config.py.")
        else:
            print("✅ Rapida Engine components found.")
            self.ready = True

    def speak(self, text: str):
        print(f"Amadeus: {text}")
        if not self.ready:
            return
        try:
            quoted_text = shlex.quote(text)
            command = (f'echo {quoted_text} | "{Config.PIPER_EXE_PATH}" --model "{Config.PIPER_MODEL_PATH}" --output_file "{Config.OUTPUT_WAV_PATH}"')
            subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            data, fs = sf.read(Config.OUTPUT_WAV_PATH, dtype='float32')
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"[TTS Error: V9 Core]: {e}")

class V10_Coqui_Core_TTS(BaseTTS):
    """V10 Coqui Core: A high-quality, offline voice using Coqui TTS."""
    def __init__(self):
        super().__init__()
        print("Loading V10 Coqui Core (Natural/Offline)...")
        try:
            self.tts = TTS(Config.COQUI_TTS_MODEL, gpu=False)
            print("✅ V10 Coqui Core initialized.")
            self.ready = True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize Coqui TTS. Error: {e}")
            self.ready = False

    def _clean_text_for_tts(self, text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FAFF"  # Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+",
            flags=re.UNICODE,
        )
        cleaned_text = emoji_pattern.sub(r'', text)
        cleaned_text = re.sub(r'[\*#`]', '', cleaned_text)
        return cleaned_text.strip()

    def speak(self, text: str):
        if not self.ready:
            print("[TTS SKIPPED] V10 Core not available.")
            return
        
        try:
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text:
                print("[TTS] No speakable text after cleaning.")
                return

            print(f"[TTS] Speaking: {cleaned_text}")
            self.tts.tts_to_file(text=cleaned_text, file_path=Config.TEMP_SPEECH_PATH)
            
            wave_obj = sa.WaveObject.from_wave_file(Config.TEMP_SPEECH_PATH)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            
            os.remove(Config.TEMP_SPEECH_PATH)
        except Exception as e:
            print(f"Error in V10 TTS engine: {e}")

