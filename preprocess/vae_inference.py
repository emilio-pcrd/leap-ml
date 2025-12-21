import torch
import numpy as np
import pretty_midi
from pathlib import Path
from preprocess.midi2pianoroll import midi_to_roll

CFG = {
    "sr": 24000,
    "fps": 100,
    "chunk_sec": 10.24,
    "stride_sec": 5.0,
    "save_dir": "/creative-ml/dataset/processed_data"
}

def midi_to_chunks(midi_path, duration=None):
    """
    Convertit un fichier MIDI en chunks (b, 1, 128, 1024) pour l'inférence VAE.

    Args:
        midi_path: chemin vers le fichier MIDI
        duration: durée totale (si None, utilise la durée du MIDI)

    Returns:
        tensor de shape (num_chunks, 1, 128, 1024)
    """
    try:
        mid = pretty_midi.PrettyMIDI(midi_path)
    except:
        raise ValueError(f"Impossible de charger le MIDI: {midi_path}")

    # Calculer la durée totale
    if duration is None:
        duration = mid.get_end_time()

    # Calculer le nombre de chunks
    num_chunks = int((duration - CFG['chunk_sec']) // CFG['stride_sec']) + 1

    chunks = []
    for i in range(num_chunks):
        start_t = i * CFG['stride_sec']

        piano_tensor = midi_to_roll(midi_path, start_t, CFG['chunk_sec'])
        if piano_tensor is not None:
            if len(piano_tensor.shape) == 2:
                piano_tensor = piano_tensor.unsqueeze(0)
            chunks.append(piano_tensor)

    if len(chunks) == 0:
        raise ValueError("Aucun chunk valide généré")

    # Stack en (b, 1, 128, 1024)
    return torch.stack(chunks, dim=0)


def get_triangle_window(length):
    """
    Creates a window that is 1.0 in the center and 0.0 at the edges.
    Prioritizes the center of predictions where the model is most confident.
    """
    # Simple Triangle Window
    w = np.bartlett(length)
    # Or Hanning for smoother curve: np.hanning(length)
    return w

def chunks_to_midi(chunks, velocity_threshold=0.1, min_note_len=0.05):
    """
    Reconstructs MIDI using Weighted Cross-Fading and Max-Velocity aggregation.

    Args:
        chunks: tensor (B, 1, 128, 1024)
        min_note_len: seconds. Filters out tiny blips (noise).
    """
    mid = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    num_chunks = chunks.shape[0]
    chunk_len = chunks.shape[3] # 1024

    # Calculate total duration in frames
    stride_frames = int(CFG['stride_sec'] * CFG['fps']) # 500
    total_frames = stride_frames * (num_chunks - 1) + chunk_len

    full_roll = np.zeros((128, total_frames))
    weight_sum = np.zeros((128, total_frames))

    window = get_triangle_window(chunk_len)

    for i, chunk in enumerate(chunks):
        chunk_roll = chunk.squeeze(0).cpu().numpy()

        start = i * stride_frames
        end = start + chunk_len

        full_roll[:, start:end] += chunk_roll * window[None, :]
        weight_sum[:, start:end] += window[None, :]

    # Avoid division by zero (edges might be close to 0)
    weight_sum[weight_sum < 1e-6] = 1.0
    full_roll = full_roll / weight_sum

    # Scale back to 0-127
    full_roll = full_roll * 127.0

    # We iterate over pitches to find connected components (notes)
    for pitch in range(128):
        velocity_curve = full_roll[pitch, :]

        # Binarize based on threshold
        is_active = velocity_curve > (velocity_threshold * 127)

        diff = np.diff(is_active.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for onset, offset in zip(starts, ends):
            duration = (offset - onset) / CFG['fps']

            if duration < min_note_len:
                continue

            segment_velocities = velocity_curve[onset:offset]
            max_vel = np.max(segment_velocities)
            midi_vel = int(np.clip(max_vel, 1, 127))

            # Create Note
            note = pretty_midi.Note(
                velocity=midi_vel,
                pitch=pitch,
                start=onset / CFG['fps'],
                end=offset / CFG['fps']
            )
            piano.notes.append(note)

    mid.instruments.append(piano)
    return mid
