import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# The parent folder containing all your audio subfolders
AUDIO_PARENT_DIR = 'standardized_audios'
# The new main folder where these spectrogram images will be saved
OUTPUT_IMAGE_DIR = 'normal_spectrogram_images/'
SAMPLE_RATE = 16000

# --- 2. MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Automatic Normal Spectrogram Generation ---")

    # Create the main output directory if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"Created main output directory: {OUTPUT_IMAGE_DIR}")

    # Create a list of all audio files to process
    files_to_process = []
    for root, dirs, files in os.walk(AUDIO_PARENT_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print(f"Error: No audio files found in '{AUDIO_PARENT_DIR}'. Please check the path.")
        exit()

    print(f"Found {len(files_to_process)} audio files to process.")

    # Process each file with a progress bar
    for audio_path in tqdm(files_to_process, desc="Generating Spectrograms"):
        try:
            # --- Create corresponding subdirectories in the output folder ---
            relative_path = os.path.relpath(os.path.dirname(audio_path), AUDIO_PARENT_DIR)
            image_output_folder = os.path.join(OUTPUT_IMAGE_DIR, relative_path)
            
            if not os.path.exists(image_output_folder):
                os.makedirs(image_output_folder)

            # --- Generate and Save Spectrogram ---
            # Load the audio file
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

            # Create the normal spectrogram using Short-Time Fourier Transform (STFT)
            stft = librosa.stft(waveform)
            stft_db = librosa.amplitude_to_db(abs(stft))

            # Create the plot
            plt.figure(figsize=(10, 4))
            # Use y_axis='hz' for a linear frequency scale
            librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
            
            # To save a clean image with no axes or labels, we turn them off
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Define the output filename for the image
            filename_without_ext = os.path.splitext(os.path.basename(audio_path))[0]
            image_output_path = os.path.join(image_output_folder, f"{filename_without_ext}.png")

            # Save the figure
            plt.savefig(image_output_path, bbox_inches='tight', pad_inches=0)
            
            # Close the plot to free up memory
            plt.close()

        except Exception as e:
            print(f"\nCould not process file {audio_path}. Error: {e}")

    print("\n--- All normal spectrograms have been saved successfully! ---")
