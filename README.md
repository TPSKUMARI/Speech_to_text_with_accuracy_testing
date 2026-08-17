# Speech-to-Text Accuracy Tester

This project is a simple Python app for testing how accurately a speech-to-text model transcribes spoken words.

It records audio from the microphone, converts the speech to text using Whisper, and compares the transcription with the text the user actually said to calculate accuracy.

## What this project does

- Records speech from your microphone
- Uses the `faster-whisper` model to transcribe audio
- Detects the spoken text and prints live transcription
- Lets you enter the original reference sentence
- Calculates accuracy using metrics such as:
  - Word Accuracy
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - BLEU score
- Saves the result as a JSON file for review

## Project files

- `speech_to_text_WER2.py` - main project script
- `requirements.txt` - required Python packages
- `videos/` - folder for video-related files or testing assets

## Requirements

- Python 3.9+
- Microphone connected to your computer
- Required packages from `requirements.txt`

## Setup

1. Open the project folder
2. Create a virtual environment (optional but recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python speech_to_text_WER2.py
```

## How it works

1. Press Enter to start listening
2. Speak naturally into the microphone
3. Press Enter again to stop recording
4. Enter the exact sentence you intended to say
5. The script compares the expected text with the transcription and shows accuracy results

## Output

After testing, the script saves a result file such as:

```text
speech_accuracy_1721234567.json
```

This contains the spoken text, transcribed text, accuracy values, and timestamp.

## Notes

- This project is best suited for testing speech recognition quality in a local environment.
- Audio quality and microphone clarity can strongly affect the results.
- The script can use CUDA if available, otherwise it falls back to CPU.
