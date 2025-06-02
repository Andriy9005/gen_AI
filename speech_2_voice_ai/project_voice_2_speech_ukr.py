import os
import whisper
import pandas as pd
from jiwer import wer
import sys
import language_tool_python
import time

tool = language_tool_python.LanguageToolPublicAPI('uk-UA')  # <- замість LanguageTool

# 🔧 Вказати шлях до ffmpeg.exe вручну
ffmpeg_path = r"C:\Users\Andriy.Bespalyy\ffmpeg\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Завантаження моделі Whisper
model = whisper.load_model("medium")


def correct_text(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# 📁 Шлях до папки з файлами
folder_path = "C:/Users/Andriy.Bespalyy/11th_project/output/project/"
transcripts = []

# 🎧 Допустимі формати
valid_extensions = (".m4a", ".mp3", ".wav", ".flac", ".webm", ".ogg", ".aac")

# 🔁 Обробка файлів
for fname in os.listdir(folder_path):
    if fname.lower().endswith(valid_extensions):
        fpath = os.path.join(folder_path, fname)
        try:
            result = model.transcribe(fpath, language="uk")
            corrected = correct_text(result["text"])
            time.sleep(3)
            transcripts.append({"file": fname, "transcript": corrected})
            print(f"✅ {fname}: {corrected[:60]}...")
        except Exception as e:
            print(f"❌ {fname}: помилка транскрипції — {str(e)}")

# 💾 Збереження транскриптів
df = pd.DataFrame(transcripts)
df.to_csv("transcriptions_uk_corrected.csv", index=False, encoding="utf-8-sig")

# 📏 Оцінка точності (WER), якщо задано референси
ground_truths = {
    "sample.m4a": "У кожному регіоні пройшли спочатку відкриті кваліфікації, в яких могли брати участь усі бажаючі колективи, а потім і закриті відбіркові, де заздалегідь відібрані найкращі команди регіонів змагались з переможцями відкритих кваліфікацій."
}

wer_scores = []
for row in transcripts:
    fname = row["file"]
    predicted = row["transcript"]
    if fname in ground_truths:
        truth = ground_truths[fname]
        error = wer(truth, predicted)
        wer_scores.append({"file": fname, "WER": error})

df_wer = pd.DataFrame(wer_scores)
df_wer.to_csv("wer_scores_uk.csv", index=False, encoding="utf-8-sig")
