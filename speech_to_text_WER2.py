import os
import site
import glob
import threading
import time
import numpy as np
import queue
import pyaudio
import json
import difflib
import sys
import select
from collections import defaultdict

# Setup NVIDIA CUDA paths first
def setup_nvidia_paths():
    all_sites = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        all_sites.append(site.getusersitepackages())
    
    for site_pkg in all_sites:
        patterns = ["nvidia/cudnn/bin", "nvidia/cublas/bin"]
        for pattern in patterns:
            search_path = os.path.join(site_pkg, pattern)
            matches = glob.glob(search_path)
            for match in matches:
                if os.path.exists(match):
                    current_path = os.environ.get('PATH', '')
                    if match not in current_path:
                        os.environ['PATH'] = match + os.pathsep + current_path

setup_nvidia_paths()

from faster_whisper import WhisperModel

class ManualStopSpeechTester:
    def __init__(self, model_size="large-v2", device="auto"):
        # Accent detection patterns
        self.accent_patterns = {
            "american": {
                "keywords": ["awesome", "dude", "guy", "like", "totally", "yeah", "super"],
                "phonetic_markers": ["r-colored", "flat_a"],
                "name": "American"
            },
            "british": {
                "keywords": ["brilliant", "lovely", "quite", "rather", "cheers", "bloke", "mate"],
                "phonetic_markers": ["non-rhotic", "broad_a"],
                "name": "British"
            },
            "australian": {
                "keywords": ["mate", "no worries", "fair dinkum", "bloody", "crikey", "arvo"],
                "phonetic_markers": ["rising_intonation", "vowel_shift"],
                "name": "Australian"
            },
            "indian": {
                "keywords": ["actually", "very good", "excellent", "kindly", "itself", "only"],
                "phonetic_markers": ["retroflex", "syllable_timing"],
                "name": "Indian"
            },
            "canadian": {
                "keywords": ["eh", "about", "sorry", "house", "out", "aboot"],
                "phonetic_markers": ["canadian_raising", "cot_caught"],
                "name": "Canadian"
            },
            "south_african": {
                "keywords": ["just now", "now now", "shame", "hey", "lekker", "braai"],
                "phonetic_markers": ["kit_split", "bath_trap"],
                "name": "South African"
            }
        }
        
        self.detected_accent = "universal"
        self.accent_confidence = {}
        self.setup_model(model_size, device)
        self.setup_audio_optimized()
        
        # Store transcriptions for accuracy testing
        self.transcriptions = []
        self.start_time = None
        self.stop_requested = False
        
    def setup_model(self, model_size, device):
        if device == "auto":
            try:
                test_model = WhisperModel("tiny", device="cuda", compute_type="float16")
                del test_model
                device = "cuda"
                compute_type = "float16"
                print(f" CUDA ready - {model_size} model")
            except:
                device = "cpu"
                compute_type = "int8"
                print(f" CPU mode - {model_size} model")
        
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        print(" Ready for speech-to-text accuracy testing")
        
    def setup_audio_optimized(self):
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.RECORD_SECONDS = 2.0
        
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
    def detect_accent(self, text):
        """Detect accent from speech patterns"""
        text_lower = text.lower()
        current_scores = {}
        
        for accent_name, config in self.accent_patterns.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    score += 2
            
            if accent_name == "british" and ("colour" in text_lower or "favourite" in text_lower):
                score += 3
            if accent_name == "american" and ("color" in text_lower or "favorite" in text_lower):
                score += 3
            
            current_scores[accent_name] = score
        
        for accent, score in current_scores.items():
            if accent not in self.accent_confidence:
                self.accent_confidence[accent] = 0
            self.accent_confidence[accent] = (self.accent_confidence[accent] * 0.8) + (score * 0.2)
        
        if self.accent_confidence:
            best_accent = max(self.accent_confidence, key=self.accent_confidence.get)
            if self.accent_confidence[best_accent] > 0.5 and best_accent != self.detected_accent:
                self.detected_accent = best_accent
                
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate WER using edit distance"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
            
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)
    
    def calculate_character_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate CER"""
        ref_chars = list(reference.lower().replace(' ', ''))
        hyp_chars = list(hypothesis.lower().replace(' ', ''))
        
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
            
        matcher = difflib.SequenceMatcher(None, ref_chars, hyp_chars)
        return 1.0 - matcher.ratio()
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words or not hyp_words:
            return 0.0
            
        ref_counts = defaultdict(int)
        for word in ref_words:
            ref_counts[word] += 1
            
        hyp_counts = defaultdict(int) 
        for word in hyp_words:
            hyp_counts[word] += 1
            
        overlap = 0
        for word in hyp_counts:
            overlap += min(hyp_counts[word], ref_counts[word])
            
        precision = overlap / len(hyp_words) if hyp_words else 0
        bp = min(1.0, len(hyp_words) / len(ref_words)) if ref_words else 0
        
        return bp * precision
        
    def calculate_accuracy_metrics(self, expected: str, transcribed: str):
        """Calculate and display accuracy metrics"""
        if not expected.strip() or not transcribed.strip():
            print(" Cannot calculate accuracy - missing text")
            return
            
        wer = self.calculate_word_error_rate(expected, transcribed)
        cer = self.calculate_character_error_rate(expected, transcribed)
        bleu = self.calculate_bleu_score(expected, transcribed)
        
        word_accuracy = max(0, 1.0 - wer) * 100
        char_accuracy = max(0, 1.0 - cer) * 100
        
        print(f"\n" + "="*70)
        print(f" ACCURACY RESULTS")
        print(f"="*70)
        print(f"What you said:     {expected}")
        print(f"What STT heard:    {transcribed}")
        print(f"-"*70)
        print(f"Word Accuracy:       {word_accuracy:.1f}%")
        print(f"Word Error Rate:     {wer*100:.1f}%")

        print(f"="*70)
        
        # Performance feedback
        if word_accuracy >= 95:
            print(" Excellent accuracy!")
        elif word_accuracy >= 85:
            print(" Good accuracy!")
        elif word_accuracy >= 70:
            print("  Acceptable accuracy")
        else:
            print(" Poor accuracy - check audio quality")
        
        # Save results
        timestamp = int(time.time())
        results = {
            "timestamp": timestamp,
            "what_you_said": expected,
            "what_stt_heard": transcribed,
            "word_accuracy": word_accuracy,
            "character_accuracy": char_accuracy,
            "wer": wer * 100,
            "cer": cer * 100,
            "bleu_score": bleu,
            "detected_accent": self.detected_accent
        }
        
        filename = f"speech_accuracy_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Results saved to: {filename}")
        print()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_listening:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def monitor_for_stop(self):
        """Monitor for Enter key press to stop recording"""
        print(" Press ENTER in this window to stop recording and calculate accuracy")
        input()  # This will wait for Enter key
        self.stop_requested = True
        print(" Stop requested - finishing current processing...")
        
    def start_listening(self):
        """Start listening and transcribing speech naturally"""
        self.is_listening = True
        self.transcriptions = []
        self.start_time = time.time()
        self.stop_requested = False
        
        print(" Listening for natural speech...")
        print(" Speak naturally. Transcriptions will appear below:")
        print("-" * 70)
        
        # Start monitor thread for manual stop
        monitor_thread = threading.Thread(target=self.monitor_for_stop, daemon=True)
        monitor_thread.start()
        
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
        audio_buffer = []
        frames_per_buffer = int(self.RATE * self.RECORD_SECONDS / self.CHUNK)
        
        while self.is_listening and not self.stop_requested:
            try:
                # Collect audio chunks
                while len(audio_buffer) < frames_per_buffer and not self.stop_requested:
                    if not self.audio_queue.empty():
                        audio_buffer.append(self.audio_queue.get())
                    else:
                        time.sleep(0.01)
                
                if self.stop_requested:
                    break
                    
                audio_data = b''.join(audio_buffer)
                audio_buffer = []
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Process if speech detected
                if np.max(np.abs(audio_np)) > 0.02:
                    segments, info = self.model.transcribe(
                        audio_np,
                        language="en",
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=300,
                            speech_pad_ms=100,
                            max_speech_duration_s=30
                        ),
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,
                        condition_on_previous_text=True,
                        word_timestamps=True
                    )
                    
                    transcription = " ".join([segment.text.strip() for segment in segments])
                    
                    if transcription and len(transcription.strip()) > 1:
                        language_prob = info.language_probability if hasattr(info, 'language_probability') else 0
                        
                        if language_prob > 0.7 or info.language == "en":
                            print(f"📝 [{time.strftime('%H:%M:%S')}] {transcription}")
                            
                            # Store transcription
                            self.transcriptions.append(transcription)
                            
                            # Run accent detection
                            self.detect_accent(transcription)
                            
            except Exception as e:
                print(f"  Processing error: {e}")
                continue
        
        # Stop listening
        self.is_listening = False
        
    def stop_listening_and_test_accuracy(self):
        """Stop listening and test accuracy"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        
        print("\n Recording stopped")
        
        if not self.transcriptions:
            print("❌ No speech detected. Try speaking louder or closer to microphone.")
            return
        
        # Combine all transcriptions
        full_transcription = " ".join(self.transcriptions).strip()
        
        print(f"\n ACCURACY TESTING")
        print(f" Complete transcription: {full_transcription}")
        print(f"-" * 70)
        
        # Ask user what they actually said
        print("Now tell me what you actually said:")
        actual_text = input("Enter what you spoke: ").strip()
        
        if actual_text:
            # Calculate accuracy
            self.calculate_accuracy_metrics(actual_text, full_transcription)
        else:
            print("❌ No reference text provided - cannot calculate accuracy")

def main():
    """Main function - manual stop speech accuracy testing"""
    print(" Manual Stop Speech-to-Text Accuracy Tester")
    print("="*55)
    print("How to use:")
    print("1. Press Enter to start recording")
    print("2. Speak naturally") 
    print("3. Press ENTER again to stop recording")
    print("4. Enter what you actually said")
    print("5. Get accuracy results!")
    print("="*55)
    
    input("Press Enter when ready to start...")
    
    print(f"\n Initializing STT model...")
    
    # Initialize STT
    stt = ManualStopSpeechTester(model_size="large-v2", device="auto")
    
    # Start listening
    stt.start_listening()
    
    # Get accuracy results
    stt.stop_listening_and_test_accuracy()
    
    print("\n Testing completed!")

if __name__ == "__main__":
    main()