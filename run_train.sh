#!/usr/bin/env bash

## On many systems you will need to set up cuda:
#module load cuda
#conda activate tf-gpu
#export PYTHONUNBUFFERED=1

latent8=8
latent30=30
gen_size=1048576

# ====== simple VAE from GPSA ======

size=10K

echo -e "\n\nTrain"
python vaes.py simpleVAE_10k train sVAE 6M.exper_10k.seqs 7 250
echo -e "\n\nPlot latent"
python vaes.py simpleVAE_10k plot_latent 6M.exper_10k.seqs
echo -e "\n\nGen 6M"
python vaes.py simpleVAE_10k gen 10000 -o gen_simpleVAE_10k


# ====== DeepSequence ======

echo -e "\n\nTrain"
python vaes.py deepsequence_latent8 train DeepSequence 6M.exper_10k.seqs 8 --epoch 12
echo -e "\n\nPlot latent"
python vaes.py deepsequence_latent8 plot_latent 6M.exper_10k.seqs
echo -e "\n\nGen 6M"
python vaes.py deepsequence_latent8 gen 10000 -o gen_deepsequence_latent8


echo -e "\n\nTrain"
python vaes.py deepsequence_latent30 train DeepSequence 6M.exper_10k.seqs 30 --epoch 32
echo -e "\n\nPlot latent"
python vaes.py deepsequence_latent30 plot_latent 6M.exper_10k.seqs
echo -e "\n\nGen 6M"
python vaes.py deepsequence_latent30 gen 10000 -o gen_deepsequence_latent30
