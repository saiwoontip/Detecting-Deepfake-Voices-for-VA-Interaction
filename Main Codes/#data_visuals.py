import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---

AUDIO_FILE_PATH = 'Codes/output_audio/fake/deepfake_gemini/deepfake_geminiS13.wav'

SAMPLE_RATE = 16000
# --- END CONFIGURATION ---


def analyze_and_visualize_audio(file_path):
    """
    Loads a single audio file and generates a dashboard of feature visualizations,
    displaying each plot one by one.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Analyzing file: {os.path.basename(file_path)} ---")


    try:
        # 1. Load the audio file
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # --- 2. Create and show plots one by one ---

        # Plot 0: Raw Audio Signal (Waveform)
        print("Displaying: Raw Audio Signal (Waveform)")
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(waveform, sr=sr)
        plt.title(f'Raw Audio Signal (Waveform)\n{os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()

        # Plot 1: Spectrogram
        print("Displaying: Spectrogram")
        plt.figure(figsize=(10, 5))
        stft = librosa.stft(waveform)
        stft_db = librosa.amplitude_to_db(abs(stft))
        librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram (Hz Scale)\n{os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

        # Plot 2: Mel Spectrogram
        print("Displaying: Mel Spectrogram")
        plt.figure(figsize=(10, 5))
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram\n{os.path.basename(file_path)}')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.show()

        # Plot 3 & 4: Harmonics and Percussive Components
        print("Displaying: Harmonic and Percussive Components")
        y_harmonic, y_percussive = librosa.effects.hpss(waveform)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y_harmonic, sr=sr, alpha=0.7, color='b')
        plt.title(f'Harmonic Component\n{os.path.basename(file_path)}')
        plt.subplot(2, 1, 2)
        librosa.display.waveshow(y_percussive, sr=sr, alpha=0.7, color='r')
        plt.title('Percussive Component')
        plt.tight_layout()
        plt.show()

        # Plot 5: Spectral Centroid
        print("Displaying: Spectral Centroid")
        plt.figure(figsize=(10, 5))
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.plot(librosa.times_like(spectral_centroid), spectral_centroid, color='w', linewidth=1.5, label='Spectral Centroid')
        plt.title(f'Spectral Centroid\n{os.path.basename(file_path)}')
        plt.colorbar(format='%+2.0f dB')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Plot 6: Spectral Rolloff
        print("Displaying: Spectral Rolloff")
        plt.figure(figsize=(10, 5))
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.plot(librosa.times_like(spectral_rolloff), spectral_rolloff, color='w', linewidth=1.5, label='Spectral Rolloff')
        plt.title(f'Spectral Rolloff\n{os.path.basename(file_path)}')
        plt.colorbar(format='%+2.0f dB')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        # Plot 7: Mel-Frequency Cepstral Coefficients (MFCCs)
        print("Displaying: MFCCs")
        plt.figure(figsize=(10, 5))
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCCs\n{os.path.basename(file_path)}')
        plt.tight_layout()
        plt.show()
        
        # Plot 8: Chroma Frequencies
        print("Displaying: Chroma Frequencies")
        plt.figure(figsize=(10, 5))
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chroma Frequencies\n{os.path.basename(file_path)}')
        plt.tight_layout()
        plt.show()

        print("\n--- All visualizations complete. ---")

    except Exception as e:
        print(f"An error occurred: {e}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    analyze_and_visualize_audio(AUDIO_FILE_PATH)