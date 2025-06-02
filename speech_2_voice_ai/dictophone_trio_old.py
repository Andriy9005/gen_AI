import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pandas as pd
from jiwer import wer
import language_tool_python
import time
import torch
from transformers import MarianMTModel, MarianTokenizer
import time

# 📍 Додати ffmpeg до шляху (якщо локально)
ffmpeg_path = r"C:\\Users\\Andriy.Bespalyy\\ffmpeg\\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 🎙️ Налаштування аудіо
duration = 10
sample_rate = 44100
device_index = 8
output_file = "mic_recording.wav"

print(f"🎤 Recording will start in 2 seconds...")
time.sleep(2)  # ⏳ Пауза 2 секунди

print(f"🎤 Recording for {duration} seconds from device #{device_index}...")
recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='int16',
    device=device_index
)
sd.wait()
write(output_file, sample_rate, recording)
print(f"✅ Audio saved as {output_file}")

# 🎧 Відтворення
sd.play(recording, samplerate=sample_rate)
sd.wait()

# 🧠 Завантаження моделі Whisper
model = whisper.load_model("medium")

# 🧠 Визначення мови автоматично
print("🧠 Detecting language...")

lang_result = model.transcribe(output_file, task="transcribe", language=None)

# ✨ Витягуємо визначену Whisper мову
raw_lang = lang_result.get("language", "unknown")

# ✅ Якщо не англійська — вважаємо українською
detected_lang = "en" if raw_lang == "en" else "uk"

print(f"🌐 Detected language (forced logic): {detected_lang}")

# Підключаємо правильний LanguageTool
if detected_lang == "uk":
    tool = language_tool_python.LanguageToolPublicAPI('uk-UA')
elif detected_lang == "en":
    tool = language_tool_python.LanguageToolPublicAPI('en-US')
else:
    print(f"⚠️ Unsupported language: {detected_lang}. Skipping grammar correction.")
    tool = None

# ✍️ Фінальна транскрипція з відомою мовою
print("📝 Transcribing...")
result = model.transcribe(output_file, language=detected_lang)
raw_text = result["text"]

# 🛠 Граматична корекція
def correct_text(text):
    if tool is None:
        return text
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

corrected = correct_text(raw_text)
print(f"✅ Corrected transcript:\n{corrected}")

# 🌐 Переклад моделлю MarianMT
if detected_lang == "uk":
    trans_model_path = "C:/Users/Andriy.Bespalyy/Desktop/models/marianmt-uk-en-hplt-final/marianmt-uk-en-hplt-final"
elif detected_lang == "en":
    trans_model_path = "C:/Users/Andriy.Bespalyy/Desktop/models/marianmt-en-uk-hplt-final/marianmt-en-uk-hplt-final"
else:
    trans_model_path = None

if trans_model_path:
    tokenizer = MarianTokenizer.from_pretrained(trans_model_path)
    translator = MarianMTModel.from_pretrained(trans_model_path)
    inputs = tokenizer([corrected], return_tensors="pt", padding=True, truncation=True).to(translator.device)
    with torch.no_grad():
        translated = translator.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"🌍 Translated text:\n{translated_text}")
else:
    translated_text = "(no translation)"

# 💾 Збереження в CSV
transcripts = [{"file": output_file, "transcript": corrected, "translation": translated_text}]
df = pd.DataFrame(transcripts)
df.to_csv("transcriptions_corrected_translated.csv", index=False, encoding="utf-8-sig")
print("📄 Saved to transcriptions_corrected_translated.csv")

# 📏 WER (якщо ground-truth доступний)
ground_truths = {
    "mic_recording.wav": "Очікуваний текст тут для оцінки точності"
}

if output_file in ground_truths:
    gt = ground_truths[output_file]
    print(f"🔍 Ground truth: {gt}")
    print(f"📝 Predicted   : {corrected}")
    error = wer(gt, corrected)
    df_wer = pd.DataFrame([{"file": output_file, "WER": error}])
    df_wer.to_csv("wer_scores.csv", index=False, encoding="utf-8-sig")
    print(f"📉 WER saved: {error:.2%}")
else:
    print("ℹ️ Ground-truth not provided → WER not calculated.")




# 📁 Шлях до CSV-файлу
csv_file = "transcriptions_corrected_translated.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 🧠 Визначення мови просто з тексту
from langdetect import detect

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"



import os
import csv
import torch
import sounddevice as sd
from scipy.io.wavfile import write

from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# 🎯 Визначення пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Завантаження моделей...")

# Англійські моделі (можна залишити, якщо потрібні)
fastpitch_en = FastPitchModel.from_pretrained("tts_en_fastpitch").to(device)
hifigan_en = HifiGanModel.from_pretrained("tts_en_hifigan").to(device)

# Українські моделі з Hugging Face
fastpitch_uk = FastPitchModel.from_pretrained("theodotus/tts_uk_fastpitch").to(device)
hifigan_uk = HifiGanModel.from_pretrained("theodotus/tts_uk_hifigan").to(device)

# 🔊 Синтез + програвання
def synthesize_and_play(text, path, lang_code="en"):
    if lang_code == "en":
        fastpitch, hifigan = fastpitch_en, hifigan_en
    else:
        fastpitch, hifigan = fastpitch_uk, hifigan_uk

    sample_rate = 22050  # ✅ Частота HiFi-GAN моделей
    with torch.no_grad():
        parsed = fastpitch.parse(text)
        spectrogram = fastpitch.generate_spectrogram(tokens=parsed)
        audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)
        audio_np = audio[0].cpu().numpy()

    write(path, sample_rate, audio_np)
    print(f"✅ Збережено у {path}")
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()

# 📄 Обробка CSV
csv_file = "transcriptions_corrected_translated.csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        text = row.get("translation", "").strip()
        if not text:
            continue
        try:
            lang_code = detect(text)
        except:
            lang_code = "en"
        lang_code = "uk" if lang_code == "uk" else "en"
        filename = os.path.join(output_dir, f"translated_{i+1}.wav")
        print(f"🔹 Озвучення ({lang_code.upper()}): {text}")
        synthesize_and_play(text, filename, lang_code=lang_code)
        break  # 🔁 Тільки перший рядок
 