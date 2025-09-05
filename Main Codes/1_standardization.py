"""
Standardize every .wav / .mp3 under output_audio/ into standardized_audios/
16 kHz, mono, –20 dBFS
"""

import os
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import traceback

# --------------------------------------------------
# 1. CONFIGURATION  (EDIT THESE TWO LINES ONLY)
# --------------------------------------------------
INPUT_DIR   = Path("Codes\output_audio")                     # <— EDIT HERE (raw files)
FFMPEG_PATH = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"              # <— EDIT HERE (full path to ffmpeg.exe)

OUTPUT_DIR  = Path("standardized_audios")              # where clean files go
TARGET_SR   = 16000           # Hz
TARGET_CH   = 1               # mono
TARGET_DB   = -20.0           # dBFS

# --------------------------------------------------
# 2. MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("--- Starting Audio Standardization ---")

    # Tell pydub where FFmpeg is
    AudioSegment.converter = FFMPEG_PATH

    # Ensure output folder exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    # Collect all audio files
    files = list(INPUT_DIR.rglob("*.wav")) + list(INPUT_DIR.rglob("*.mp3"))
    if not files:
        print("⚠️  No .wav or .mp3 files found in", INPUT_DIR.resolve())
        exit()

    print(f"Found {len(files)} files to process.\n")

    # Process with progress bar
    for src in tqdm(files, desc="Standardizing"):
        try:
            rel = src.relative_to(INPUT_DIR)
            dst = (OUTPUT_DIR / rel).with_suffix(".wav")
            dst.parent.mkdir(parents=True, exist_ok=True)

            audio = AudioSegment.from_file(str(src))
            audio = audio.set_frame_rate(TARGET_SR)
            audio = audio.set_channels(TARGET_CH)
            gain  = TARGET_DB - audio.dBFS
            audio = audio.apply_gain(gain)
            audio.export(str(dst), format="wav", parameters=["-acodec", "pcm_s16le"])

        except Exception:
            print(f"\n❌ Error on {src}")
            traceback.print_exc()

    print("\n✅ Done!  All clean files are in", OUTPUT_DIR.resolve())