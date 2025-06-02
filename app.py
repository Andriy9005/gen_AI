from flask import Flask, render_template, jsonify
import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from jiwer import wer
import language_tool_python
from langdetect import detect
from transliterate import translit
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

app = Flask(__name__)

# ⏱️ Основні параметри
duration = 10
sample_rate = 44100
device_index = 1
output_file = "mic_recording.wav"
ffmpeg_path = r"C:\\Users\\Andriy.Bespalyy\\ffmpeg\\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 📥 Завантаження моделей
whisper_model = whisper.load_model("medium")
fastpitch_en = FastPitchModel.from_pretrained("tts_en_fastpitch").cuda()
hifigan_en = HifiGanModel.from_pretrained("tts_en_hifigan").cuda()
fastpitch_uk = FastPitchModel.from_pretrained("theodotus/tts_uk_fastpitch").cuda()
hifigan_uk = HifiGanModel.from_pretrained("theodotus/tts_uk_hifigan").cuda()

def transliterate_uk(text):
    try:
        return translit(text, 'uk', reversed=True)
    except Exception:
        return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    try:
        sd.default.device = device_index
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        write(output_file, sample_rate, recording)

        # 🎤 Whisper + мова
        lang_result = whisper_model.transcribe(output_file, language=None)
        raw_lang = lang_result.get("language", "unknown")
        detected_lang = "en" if raw_lang == "en" else "uk"

        tool = None
        if detected_lang == "uk":
            tool = language_tool_python.LanguageToolPublicAPI('uk-UA')
        elif detected_lang == "en":
            tool = language_tool_python.LanguageToolPublicAPI('en-US')

        result = whisper_model.transcribe(output_file, language=detected_lang)
        raw_text = result["text"]

        def correct_text(text):
            if tool is None:
                return text
            try:
                matches = tool.check(text)
                return language_tool_python.utils.correct(text, matches)
            except:
                return text

        corrected = correct_text(raw_text)

        # 🔄 Переклад
        if detected_lang == "uk":
            trans_model_path = "C:/marianmt-uk-en-hplt-final/marianmt-uk-en-hplt-final"
        else:
            trans_model_path = "C:/marianmt-en-uk-hplt-final/marianmt-en-uk-hplt-final"

        tokenizer = MarianTokenizer.from_pretrained(trans_model_path)
        translator = MarianMTModel.from_pretrained(trans_model_path)
        inputs = tokenizer([corrected], return_tensors="pt", padding=True, truncation=True).to(translator.device)
        with torch.no_grad():
            translated = translator.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        # 🔊 Озвучення
        lang_code = detect(translated_text)
        lang_code = "uk" if lang_code == "uk" else "en"
        text = transliterate_uk(translated_text) if lang_code == "uk" else translated_text
        fastpitch = fastpitch_uk if lang_code == "uk" else fastpitch_en
        hifigan = hifigan_uk if lang_code == "uk" else hifigan_en

        fastpitch.eval()
        with torch.no_grad():
            parsed = fastpitch.parse(text)
            spectrogram = fastpitch.generate_spectrogram(tokens=parsed, speaker=0 if lang_code == "uk" else None)
            audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)
            audio_np = audio[0].cpu().numpy()

        filename = f"static/translated.wav"
        write(filename, 22050, audio_np)

        return jsonify({
            "original": corrected,
            "translated": translated_text,
            "audio": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
