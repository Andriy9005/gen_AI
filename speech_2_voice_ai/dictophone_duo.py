import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pandas as pd
from jiwer import wer
import language_tool_python
import time

# 📍 Додати ffmpeg до шляху (якщо локально)
ffmpeg_path = r"C:\Users\Andriy.Bespalyy\ffmpeg\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# 🎙️ Налаштування аудіо
duration = 10
sample_rate = 44100
device_index = 8
output_file = "mic_recording.wav"

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
detected_lang = lang_result["language"]
print(f"🌐 Detected language: {detected_lang}")

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

# 💾 Збереження в CSV
transcripts = [{"file": output_file, "transcript": corrected}]
df = pd.DataFrame(transcripts)
df.to_csv("transcriptions_corrected.csv", index=False, encoding="utf-8-sig")
print("📄 Saved to transcriptions_corrected.csv")

# 📏 WER (якщо ground-truth доступний)
ground_truths = {
    "mic_recording.wav": "Очікуваний текст тут для оцінки точності"  # або англійський, якщо аудіо було англійською
}

wer_scores = []
if output_file in ground_truths:
    gt = ground_truths[output_file]
    print(f"🔍 Ground truth: {gt}")
    print(f"📝 Predicted   : {corrected}")
    error = wer(gt, corrected)
    wer_scores.append({"file": output_file, "WER": error})
    df_wer = pd.DataFrame(wer_scores)
    df_wer.to_csv("wer_scores.csv", index=False, encoding="utf-8-sig")
    print(f"📉 WER saved: {error:.2%}")
else:
    print("ℹ️ Ground-truth not provided → WER not calculated.")
