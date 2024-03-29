#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

seq_file=/Users/yuxuan/projects/msa/seq/train.seqs
gen_size=1048576

# ====== simple VAE from GPSA ======

# echo -e "\n\n===== Train"
# python vae_keras.py sVAE train sVAE $seq_file 7 250 --epoch 5
echo -e "\n\n===== Plot latent"
python vae_keras.py sVAE plot_latent $seq_file
echo -e "\n\n===== Generating sequences"
python vae_keras.py sVAE gen $gen_size -o ../seq/gen_sVAE.seqs


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

# echo -e "\n\nTrain"
# python vae_keras.py deepVAE train Deep_VAE $seq_file 8 5 --epoch 6
# echo -e "\n\nPlot latent"
# python vae_keras.py deepVAE plot_latent $seq_file
# echo -e "\n\n===== Generating sequences"
# python vae_keras.py deepVAE gen $gen_size -o ../seq/gen_dVAE.seqs