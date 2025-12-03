# voice_handler_test.py
# A standalone script to test the complete STT -> TTS voice loop.

import speech_recognition as sr
import threading
from faster_whisper import WhisperModel
import io
import os
import simpleaudio as sa
from TTS.api import TTS
import re
import time
import numpy as np

# --- Initialize Coqui TTS ---
# This is done once at the start to avoid reloading the model.
print("Initializing Coqui TTS engine... (This may take a moment)")
try:
    tts = TTS("tts_models/en/ljspeech/vits", gpu=False)
    print("✅ Coqui TTS engine initialized.")
    tts_ready = True
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not initialize Coqui TTS. Error: {e}")
    print("Please ensure you have run 'pip install TTS' and that you have a working internet connection for the first run to download the model.")
    tts_ready = False
# -----------------------------

class VoiceHandler:
    def __init__(self):
        print("Initializing Voice Handler with Faster Whisper...")
        try:
            # Using a small, efficient model for testing
            model_size = "base.en"
            self.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 1000 # Adjust this based on your mic sensitivity
            self.recognizer.dynamic_energy_threshold = True
            print("✅ Voice Handler initialized.")
            self.stt_ready = True
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not initialize Whisper STT. Error: {e}")
            self.stt_ready = False

    def _clean_text_for_tts(self, text):
        """Removes emojis and other non-speech characters to prevent errors."""
        # This regex is now more comprehensive
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
        # Also remove common markdown characters that can cause issues
        cleaned_text = re.sub(r'[\*#`]', '', cleaned_text)
        return cleaned_text.strip()

    def speak(self, text):
        """Converts text to an audio file and plays it in a separate thread."""
        if not tts_ready:
            print("[TTS SKIPPED] Engine not available.")
            return
        # Run the speaking part in its own thread to avoid blocking
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        """Private method to generate and play speech."""
        try:
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text:
                print("[TTS] No speakable text after cleaning.")
                return

            print(f"[TTS] Speaking: {cleaned_text}")
            # Use a temporary file to store the speech output
            output_wav_path = "temp_speech.wav"
            tts.tts_to_file(text=cleaned_text, file_path=output_wav_path)
            
            # Play the audio file using simpleaudio
            wave_obj = sa.WaveObject.from_wave_file(output_wav_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            
            # Clean up the temporary file
            os.remove(output_wav_path)
        except Exception as e:
            print(f"Error in TTS engine: {e}")

    def listen(self, callback):
        """Starts the listening process in a separate thread."""
        if not self.stt_ready:
            print("[STT SKIPPED] Engine not available.")
            return
        threading.Thread(target=self._listen_thread, args=(callback,), daemon=True).start()

    def _listen_thread(self, callback):
        """Private method for the listening thread."""
        recognized_text = ""
        with sr.Microphone() as source:
            print("\nListening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Listen for speech with a timeout and phrase limit
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                print("Recognizing speech...")
                
                # Convert audio to a format Whisper can process
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe the audio
                segments, _ = self.stt_model.transcribe(audio_data, beam_size=5)
                recognized_text = "".join(segment.text for segment in segments).strip()
                
            except sr.WaitTimeoutError:
                recognized_text = "[SILENCE] No speech detected."
            except Exception as e:
                recognized_text = f"[ERROR] Could not recognize: {e}"
        
        # Pass the result (or error) to the callback function
        callback(recognized_text)

# --- Main Test Loop ---
def main():
    """Function to run the interactive test."""
    # A flag to keep the main loop running
    is_running = True
    
    handler = VoiceHandler()

    # Check if the core components initialized correctly
    if not tts_ready or not handler.stt_ready:
        print("\nOne or more core voice components failed to initialize. Exiting.")
        return

    def process_input(text: str):
        """The callback function that handles the recognized text."""
        nonlocal is_running
        print(f"You said: {text}")

        if "exit" in text.lower() or "goodbye" in text.lower():
            print("Exit command received. Shutting down.")
            handler.speak("Goodbye, Lucifer.")
            time.sleep(2) # Give time for the final message to play
            is_running = False
            return
        
        # Echo the recognized text back
        if not text.startswith("["): # Don't echo errors/silence
            handler.speak(f"You said... {text}")
        
        # A small delay to prevent immediate re-listening while TTS is speaking
        time.sleep(1)
        
        # If still running, trigger the next listen
        if is_running:
            handler.listen(process_input)

    # --- Start the conversation ---
    handler.speak("Voice handler test initiated. Say something to begin, or say 'exit' to stop.")
    handler.listen(process_input)

    # Keep the main script alive while the threads do their work
    while is_running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested.")
            is_running = False
    
    print("Test finished.")


if __name__ == "__main__":
    main()
