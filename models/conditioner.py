import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


def note_count_module(roll, min_len=20):
    """
    Counts valid notes in a piano roll batch based on duration.

    Args:
        roll: Tensor of shape (B, 1, 128, T) or (1, 128, T)
              Values should be > 0 for active notes.
        min_len: Minimum length (in frames) to count as a valid note.

    Returns:
        counts: Tensor of shape (B, 1) containing the note count for each sample.
    """
    # If input is just (128, T), unsqueeze to (1, 1, 128, T)
    if roll.dim() == 2:
        roll = roll.unsqueeze(0).unsqueeze(0)
    elif roll.dim() == 3:
        roll = roll.unsqueeze(1)

    B, C, P, T = roll.shape
    device = roll.device

    # Initialize output
    batch_counts = torch.zeros((B, 1), device=device, dtype=torch.float32)

    # Iterate over batch
    for b in range(B):
        sample_count = 0

        # Iterate over pitches (128)
        for p in range(P):
            # Shape (T,)
            active = roll[b, 0, p] > 0

            # We prepend 0 to detect if a note starts immediately at t=0
            diff = torch.diff(active.int(), prepend=torch.tensor([0], device=device))

            starts = torch.where(diff == 1)[0]
            ends = torch.where(diff == -1)[0]

            if len(ends) < len(starts):
                ends = torch.cat([ends, torch.tensor([T], device=device)])

            durations = ends - starts
            valid_notes = (durations >= min_len).sum()

            sample_count += valid_notes

        batch_counts[b] = sample_count

    return batch_counts



class AudioConditioner(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", output_dim=768, freeze=True):
        super().__init__()

        # pre-trained model (MERT)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def process_audio(self, audio_waveform, sampling_rate=24000):
        """
        audio_waveform: (Batch, Time) numpy or tensor
        """
        inputs = self.processor(
            audio_waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sampling_rate * 10 # Assuming 10s chunks
        )
        return inputs

    def forward(self, input_values):
        """
        input_values: input tensor from processor
        Returns: (Batch, Seq_Len, Feature_Dim)
        """
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        features = outputs.last_hidden_state

        return features


if __name__ == "__main__":
    conditioner = AudioConditioner()

    test_audio = [torch.randn(80_000).numpy(), torch.randn(80_000).numpy()]
    inputs = conditioner.process_audio(test_audio, sampling_rate=24000)


    input_tensor = inputs['input_values']
    print(input_tensor.shape)
    feats = conditioner(input_tensor)
    print(f"Audio Features Shape: {feats.shape}")
