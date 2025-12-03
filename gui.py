# gui.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Graphical User Interface
#   Description: Manages the customtkinter GUI, event handling, and voice loop.
# =================================================================================================

import threading
import customtkinter as ctk
import speech_recognition as sr
import numpy as np
from faster_whisper import WhisperModel

from config import Config
from core_logic import get_amadeus_response

class AmadeusGUI(ctk.CTk):
    """The main graphical user interface for Project Amadeus."""
    def __init__(self, tts_engine, memory_manager):
        super().__init__()
        self.tts_engine = tts_engine
        self.memory_manager = memory_manager
        
        # --- Voice Loop Components ---
        self.voice_thread = None
        self.voice_running = threading.Event()
        self.recognizer = sr.Recognizer()
        self.whisper_model = None # Lazily loaded
        
        self._setup_ui()

    def _setup_ui(self):
        """Initializes all UI components."""
        self.title("Amadeus - Evolved Core (v3.0 Modular)")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.chat_frame = ctk.CTkTextbox(self, state="disabled", wrap="word", font=("Arial", 14))
        self.chat_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        self.input_entry = ctk.CTkEntry(self, placeholder_text="Talk to Amadeus...", font=("Arial", 14))
        self.input_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.input_entry.bind("<Return>", self.send_message)

        self.send_button = ctk.CTkButton(self, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=(0, 10))
        
        self.voice_button = ctk.CTkButton(self, text="Start Voice", command=self.toggle_voice_mode)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.status_label = ctk.CTkLabel(self, text="Status: Idle", font=("Arial", 12))
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

    def update_status(self, message):
        """Updates the status bar text safely from any thread."""
        self.status_label.configure(text=f"Status: {message}")

    def add_message(self, sender, message):
        """Adds a message to the chat window."""
        self.chat_frame.configure(state="normal")
        self.chat_frame.insert("end", f"{sender}: {message}\n\n")
        self.chat_frame.configure(state="disabled")
        self.chat_frame.see("end")

    def send_message(self, event=None):
        """Handles sending a text message."""
        prompt = self.input_entry.get()
        if not prompt:
            return
        self.add_message("Lucifer", prompt)
        self.input_entry.delete(0, "end")
        
        # Run AI response in a separate thread to keep UI responsive
        threading.Thread(target=self.get_and_display_response, args=(prompt,)).start()

    def get_and_display_response(self, prompt, is_voice=False):
        """Gets response from AI, displays it, and speaks it if in voice mode."""
        response = get_amadeus_response(prompt, self.memory_manager, self.update_status)
        self.add_message("Amadeus", response)
        self.update_status("Idle")
        
        if is_voice and self.tts_engine.ready:
            self.tts_engine.speak(response)

    def toggle_voice_mode(self):
        """Starts or stops the voice recognition loop."""
        if self.voice_running.is_set():
            self.voice_running.clear()
            if self.voice_thread:
                self.voice_thread.join()
            self.voice_button.configure(text="Start Voice")
            self.update_status("Idle")
        else:
            self.voice_running.set()
            self.voice_thread = threading.Thread(target=self.run_voice_loop, daemon=True)
            self.voice_thread.start()
            self.voice_button.configure(text="Stop Voice")

    def run_voice_loop(self):
        """The main loop for listening to and processing voice commands."""
        # --- Lazy Loading of Whisper Model ---
        if not self.whisper_model:
            self.update_status("Loading speech recognition model...")
            print("[Voice] Loading Whisper model for the first time...")
            self.whisper_model = WhisperModel(Config.WHISPER_MODEL, device="cpu", compute_type="int8")
            print("[Voice] Whisper model loaded.")
        
        if self.tts_engine.ready:
            self.tts_engine.speak("Voice mode activated.")

        while self.voice_running.is_set():
            try:
                with sr.Microphone(device_index=Config.MICROPHONE_INDEX, sample_rate=16000) as source:
                    self.update_status("Calibrating to ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                    self.update_status("Listening...")
                    audio = self.recognizer.listen(source)
                
                self.update_status("Recognizing speech...")
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if np.abs(audio_np).mean() < 0.01: # Basic silence detection
                    print("[Voice Loop] Silence detected, skipping.")
                    continue

                segments, _ = self.whisper_model.transcribe(audio_np, beam_size=5)
                command = "".join(segment.text for segment in segments).strip()

                if command:
                    self.add_message("Lucifer (Voice)", command)
                    # The response generation and speaking are handled in this thread
                    self.get_and_display_response(command, is_voice=True)
                else:
                    self.update_status("Idle")

            except sr.UnknownValueError:
                print("[Voice Loop] Could not understand audio.")
                self.update_status("Idle")
            except Exception as e:
                print(f"An error occurred in the voice loop: {e}")
                self.update_status(f"Error: {e}")
        
        if self.tts_engine.ready:
            self.tts_engine.speak("Voice mode deactivated.")
