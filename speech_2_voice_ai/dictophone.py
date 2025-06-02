import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pandas as pd
from jiwer import wer
import language_tool_python
import time

# ⚙️ Whisper + LanguageTool
tool = language_tool_python.LanguageToolPublicAPI('uk-UA')
model = whisper.load_model("medium")

# 📍 Вказати шлях до ffmpeg, якщо не в системі
ffmpeg_path = r"C:\Users\Andriy.Bespalyy\ffmpeg\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

import sounddevice as sd
from scipy.io.wavfile import write

duration = 10  # seconds
sample_rate = 44100
device_index = 8  # 🎯 Це ваш активний мікрофон
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

# 🎧 Відтворення для перевірки
sd.play(recording, samplerate=sample_rate)
sd.wait()



# 🧠 Транскрипція
print("🧠 Transcribing...")
result = model.transcribe(output_file, language="uk")
raw_text = result["text"]

# 📝 Граматична корекція
def correct_text(text):
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

corrected = correct_text(raw_text)
print(f"✅ Corrected transcript:\n{corrected}")

# 💾 Збереження в CSV
transcripts = [{"file": output_file, "transcript": corrected}]
df = pd.DataFrame(transcripts)
df.to_csv("transcriptions_uk_corrected.csv", index=False, encoding="utf-8-sig")
print("📄 Saved to transcriptions_uk_corrected.csv")

# 📏 WER (опціонально)
ground_truths = {
    "mic_recording.wav": "Очікуваний текст тут для оцінки точності"  # Заміни на свій еталон
}

wer_scores = []
if output_file in ground_truths:
    gt = ground_truths[output_file]
    error = wer(gt, corrected)
    wer_scores.append({"file": output_file, "WER": error})
    df_wer = pd.DataFrame(wer_scores)
    df_wer.to_csv("wer_scores_uk.csv", index=False, encoding="utf-8-sig")
    print(f"📉 WER saved: {error:.2%}")
else:
    print("ℹ️ Ground-truth not provided → WER not calculated.")
