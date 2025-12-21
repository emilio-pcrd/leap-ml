import os
import torch
import numpy as np
import librosa
import pretty_midi
from tqdm import tqdm

CFG = {
    "sr": 24000,
    "fps": 100,
    "chunk_sec": 10.24,
    "stride_sec": 5.0,
    "save_dir": "/mnt/ssd-samsung/creative-ml/dataset/processed_data"
}

def midi_to_roll(midi_path, start_time, duration):
    """
    Extracts a (1, 128, 1024) tensor from MIDI.
    """
    try:
        mid = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None

    num_frames = int(duration * CFG['fps'])
    roll = mid.get_piano_roll(fs=CFG['fps']) # (128, T)

    start_idx = int(start_time * CFG['fps'])
    end_idx = start_idx + num_frames

    # end of song padding
    if roll.shape[1] < end_idx:
        pad_len = end_idx - roll.shape[1]
        roll = np.pad(roll, ((0,0), (0, pad_len)))

    # Crop Time Window
    segment = roll[:, start_idx:end_idx] # (128, 1024)
    segment = segment / 127.0 # for velocities (normalize)

    return torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

def process_pair(audio_path, midi_path, file_id):
    y, _ = librosa.load(audio_path, sr=CFG['sr'], mono=True)
    total_dur = librosa.get_duration(y=y, sr=CFG['sr'])

    num_chunks = int((total_dur - CFG['chunk_sec']) // CFG['stride_sec']) + 1

    for i in range(num_chunks):
        start_t = i * CFG['stride_sec']

        start_s = int(start_t * CFG['sr'])
        end_s = int((start_t + CFG['chunk_sec']) * CFG['sr'])

        audio_chunk = y[start_s:end_s]

        expected_len = int(CFG['chunk_sec'] * CFG['sr'])
        if len(audio_chunk) < expected_len:
            audio_chunk = np.pad(audio_chunk, (0, expected_len - len(audio_chunk)))

        piano_tensor = midi_to_roll(midi_path, start_t, CFG['chunk_sec'])
        if piano_tensor is None: continue

        save_path = os.path.join(CFG['save_dir'], f"{file_id}_{i:03d}.pt")
        torch.save({
            "audio": torch.tensor(audio_chunk, dtype=torch.float32),
            "piano": piano_tensor
        }, save_path)

if __name__ == "__main__":
    os.makedirs(CFG['save_dir'], exist_ok=True)
