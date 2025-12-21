# creative-ml

## Interactive Demo
Experience the LEAP model in real-time in your browser (no installation required):
[**Launch LEAP Studio**](https://yourusername.github.io/leap-studio/)


## Projet ML - ATIAM Music Programming

This repo contains the code for the LEAP-ml project

The project is described below:

From raw wav music, the proposed model generates a **piano arrangement**, depending on a difficulty level, chosen as a parameter for the model.

It takes as input the pop audio wav song, and generate the piano roll/MIDI arrangement as a piano roll.

Here is a comprehensive pipeline of our propose approach:

- for inference:
    - slices the audio into overlapping chuncks of 10,24 seconds (e.g 0-10, 5-15, ...)
    - each chunk is pass trhough the model as a condition.
    - the diffusion transformer generate a latent code that is then decoded as a piano roll through the decoder.
    - after each chunck have been pass through the model, we concatenate them by crossfading
    - an algorithm is used to get the MIDI score from that predicted piano roll

- preprocessing:
    - audio pop songs and piano are preprocessed by the Pop2Piano preprocessing; we get audio/MIDI aligned pairs of data
    - for each data, we convert the MIDI representation into piano roll representation, and save the audio pairs to get a dataset.
    - A `dataset class` is used to sample random chuncks of equal sizes of random data to get a batch.

- training:
    - in the `dataset class`, the audio part of the extracted batch is pass through the **conditioner** (pretrained MERT model).
    - for the VAE training, the piano roll representations are used to train the model.
    It learns to generate latent code of size (B, 4, 12, 128) from piano rolls of size (B, 1, 96, 1024)
    - for the Diffusion training, the latent code are used to train a denoiser that learn to predict the noise between 2 timesteps of the diffusion. The dimensions are the same.
    - The Decoder takes as input a latent code of size (B, 4, 12, 128) and decode it to a (B, 1, 96, 1024) piano roll.
