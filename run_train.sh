#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

seq_file=seq/6M.exper.seqs
gen_size=1048576

# ====== simple VAE from GPSA ======

model_name=simpleVAE

echo -e "\n\n===== Train"
python vae_keras.py simpleVAE train sVAE $seq_file 7 250
echo -e "\n\n===== Plot latent"
python vae_keras.py simpleVAE plot_latent $seq_file
echo -e "\n\n===== Generating sequences"
python vae_keras.py simpleVAE gen $gen_size -o gen_simpleVAE_{$gen_size}


# ====== DeepSequence ======

# echo -e "\n\nTrain"
# python vae_keras.py deepsequence_latent8 train DeepSequence $seq_file 8 --epoch 12
# echo -e "\n\nPlot latent"
# python vae_keras.py deepsequence_latent8 plot_latent $seq_file
# echo -e "\n\nGen 6M"
# python vae_keras.py deepsequence_latent8 gen $gen_size -o gen_deepsequence_latent8


# echo -e "\n\nTrain"
# python vae_keras.py deepsequence_latent30 train DeepSequence $seq_file 30 --epoch 32
# echo -e "\n\nPlot latent"
# python vae_keras.py deepsequence_latent30 plot_latent $seq_file
# echo -e "\n\nGen 6M"
# python vae_keras.py deepsequence_latent30 gen $gen_size -o gen_deepsequence_latent30
