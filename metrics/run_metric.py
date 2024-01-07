#!/usr/bin/env python3
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# model_seq = "seqs/gen_sVAE.exper.seqs" 
model_seq = "seqs/deepVAE-2-629153"
# model_seq = "seqs/gen_potts_seq.1048576"

print(f"Processing r20 metrics for file \033[96m{model_seq}\033[0m")

for i in range(2, 14):
    start_time = time.time()  # Start time of the loop

    j = str(i)
    command = (
        "python tee_naive_PI.py "
        + j
        + f" 500 seqs/6M_target.seq {model_seq} seqs/gen2.indep.seq"
    )

    logging.info(f"Running command for iteration {j}")
    os.system(command)
    end_time = time.time()  # End time of the loop

    logging.info(f"Command for iteration {j} completed. Time taken: {end_time - start_time} seconds")

time.sleep(2)
os.system("python plot_comparison3.py")