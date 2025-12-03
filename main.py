# main.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Main Application Entry Point
#   Description: Initializes all components and starts the application.
# =================================================================================================

import os
import warnings
import torch
import speech_recognition as sr

# --- Universal Suppressions ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Module Imports ---
from config import Config
from core_logic import MemoryManager, initialize_core_beliefs
from tts_engines import V1_Core_TTS, V1_5_Core_TTS, V8_Core_TTS, V9_Core_TTS, V10_Coqui_Core_TTS
from gui import AmadeusGUI
import tools

def select_tts_engine():
    """
    Presents a menu to the user to select a TTS engine for the session.
    Also handles one-time voice cloning setup for the V8 Core.
    """
    print("\n--- Project Amadeus: Evolved Grand Unified Core ---")
    print("Select a voice incarnation for this session:")
    print("[1] V8 Genesis Core (Cloned, Offline)")
    print("[2] V9 Rapida Core (Fast, Offline)")
    print("[3] V10 Coqui Core (Natural, Offline)") # <-- New Option
    choice = input("Enter your choice (1-3): ")

    tts_engine = None
    if choice == '1000':
        tts_engine = V1_Core_TTS()
    elif choice == '20000':
        tts_engine = V1_5_Core_TTS()
    elif choice == '1':
        tts_engine = V8_Core_TTS()
        if tts_engine.ready:
            # --- One-Time Voice Cloning ---
            if not os.path.exists(Config.VOICE_PROFILE_FILE):
                print("\n--- VOICE CLONING (One-Time Setup) ---")
                recognizer = sr.Recognizer()
                try:
                    with sr.Microphone(device_index=Config.MICROPHONE_INDEX, sample_rate=16000) as source:
                        print("Please say: 'I am the creator of this consciousness.'")
                        recognizer.adjust_for_ambient_noise(source, duration=1)
                        audio_data = recognizer.listen(source)
                    
                    wav_filename = "genesis_voice.wav"
                    with open(wav_filename, "wb") as f:
                        f.write(audio_data.get_wav_data())
                    
                    tts_engine.speaker_embeddings = tts_engine.create_speaker_embedding(wav_filename)
                    torch.save(tts_engine.speaker_embeddings, Config.VOICE_PROFILE_FILE)
                    os.remove(wav_filename)
                    print("✅ Voice cloned and cached.")
                except Exception as e:
                    print(f"Voice cloning failed: {e}. V8 Core will not function.")
                    tts_engine.ready = False
            else:
                tts_engine.speaker_embeddings = torch.load(Config.VOICE_PROFILE_FILE)
                print("✅ Cached voice profile loaded.")
    elif choice == '2':
        tts_engine = V9_Core_TTS()
    elif choice == '3': # <-- New Block
        tts_engine = V10_Coqui_Core_TTS()

    return tts_engine

def main():
    """
    The main function to initialize and run the Amadeus application.
    """
    # 1. Initialize Core Systems
    initialize_core_beliefs()
    memory_manager = MemoryManager()
    
    # 2. Inject memory manager into the tools module
    tools.set_memory_manager(memory_manager)

    # 3. Select Voice Engine
    tts_engine = select_tts_engine()

    # 4. Launch GUI if engine is ready
    if tts_engine and tts_engine.ready:
        app = AmadeusGUI(tts_engine, memory_manager)
        app.mainloop()
    else:
        print("Selected TTS engine failed to initialize or choice was invalid. Exiting.")

if __name__ == "__main__":
    main()
