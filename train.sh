#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

seq_file=../seq/6M_train.seqs
gen_size=1048576

# ====== simple VAE from GPSA ======

# echo -e "\n\n===== Train"
# python vae_keras.py simpleVAE train sVAE $seq_file 7 250 --epoch 8
# echo -e "\n\n===== Plot latent"
# python vae_keras.py simpleVAE plot_latent $seq_file
# echo -e "\n\n===== Generating sequences"
# python vae_keras.py simpleVAE gen $gen_size -o gen_simpleVAE_$gen_size


# ====== DeepSequence ======

# echo -e "\n\nTrain"
# python vae_keras.py deepsequence_latent8 train DeepSequence $seq_file 8 --epoch 4
# echo -e "\n\nPlot latent"
# python vae_keras.py deepsequence_latent8 plot_latent $seq_file
# echo -e "\n\nGen 6M"
# python vae_keras.py deepsequence_latent8 gen $gen_size -o gen_deepsequence_latent8


# echo -e "\n\nTrain"
# python vae_keras.py deepsequence_latent30 train DeepSequence $seq_file 30 --epoch 5
# echo -e "\n\nPlot latent"
# python vae_keras.py deepsequence_latent30 plot_latent $seq_file
# echo -e "\n\nGen 6M"
# python vae_keras.py deepsequence_latent30 gen $gen_size -o gen_deepsequence_latent30


# ====== Deep_VAE from Haldane ======

echo -e "\n\nTrain"
python vae_keras.py deepVAE_Haldane_latent8 train Deep_VAE $seq_file 8 5 --epoch 4
echo -e "\n\nPlot latent"
python vae_keras.py deepVAE_Haldane_latent8 plot_latent $seq_file
echo -e "\n\n===== Generating sequences"
python vae_keras.py deepVAE_Haldane_latent8 gen $gen_size -o gen_deepVAE_Haldane_latent8