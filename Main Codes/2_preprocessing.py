import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# This should be the parent folder containing all your audio subfolders.
AUDIO_PARENT_DIR = 'standardized_audios'
OUTPUT_CSV_PATH = 'features.csv'
SAMPLE_RATE = 16000
# --- NEW: We will only load the first 10 seconds of any audio file to prevent getting stuck.
MAX_DURATION_SECONDS = 5

# --- 2. FEATURE EXTRACTION FUNCTION ---
def extract_features_for_file(file_path):
    """
    Loads an audio file and extracts a dictionary of features.
    For each feature type, it calculates the mean and standard deviation.
    """
    try:
        # --- UPDATED: Load audio file with a duration limit ---
        waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION_SECONDS)
        
        # --- Extract Features ---
        
        # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20)
        
        # 2. Chroma Frequencies
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        
        # 3. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        
        # 4. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
        
        # 5. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        
        # 6. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=waveform)
        
        # 7. Harmonics and Perceptual
        y_harmonic, y_percussive = librosa.effects.hpss(waveform)
        
        
        # --- Aggregate Features ---
        # For each feature, calculate the mean and standard deviation
        features = {
            'chroma_mean': np.mean(chroma),
            'chroma_std': np.std(chroma),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'harmonic_mean': np.mean(y_harmonic),
            'harmonic_std': np.std(y_harmonic),
            'percussive_mean': np.mean(y_percussive),
            'percussive_std': np.std(y_percussive),
        }
        
        # Add MFCCs (which have multiple coefficients)
        for i in range(len(mfcc)):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
            
        return features

    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return None

# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    print("--- Starting Detailed Audio Feature Extraction ---")
    
    all_features = []
    
    # Create a list of all files to process first
    files_to_process = []
    for root, dirs, files in os.walk(AUDIO_PARENT_DIR):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                files_to_process.append(os.path.join(root, file))

    # Process each file with a progress bar
    for full_path in tqdm(files_to_process, desc="Processing Files"):
        
        # --- UPDATED LABELING LOGIC ---
        # Determine the label based on the folder the file is in
        if 'real' in full_path or 'LJSpeech' in full_path:
            label = 'real'
        elif 'deepfake' in full_path:
            label = 'fake'
        else:
            continue 

        # Extract features for one file
        extracted_features = extract_features_for_file(full_path)
        
        if extracted_features is not None:
            # Add the filename and label to the dictionary of features
            extracted_features['filename'] = os.path.relpath(full_path, AUDIO_PARENT_DIR)
            extracted_features['label'] = label
            all_features.append(extracted_features)

    # Convert the list of dictionaries to a pandas DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Save the final DataFrame to a CSV file
    df_features.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n--- Feature extraction complete! ---")
    print(f"Successfully saved {len(df_features)} rows of features to '{OUTPUT_CSV_PATH}'.")
    print("\nLabel distribution:")
    print(df_features['label'].value_counts())
