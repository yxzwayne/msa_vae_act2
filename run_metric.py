#!/usr/bin/env python3
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

gen_model_seq = "/Users/yuxuan/projects/msa/seq/gen_esm_msa.seqs"

target_seq = "/Users/yuxuan/projects/msa/seq/6M_test.seqs"
gen_indep_seq = "/Users/yuxuan/projects/msa/seq/gen2.indep.seqs"
gen_potts_seq = "/Users/yuxuan/projects/msa/seq/gen_potts.seqs"

print(f"Processing r20 metrics for file \033[96m{gen_model_seq}\033[0m")

for i in range(2, 14):
    start_time = time.time()  # Start time of the loop

    j = str(i)
    command = (
        "python tee_naive_PI.py "
        + j
        + f" 500 {target_seq} {gen_model_seq} {gen_indep_seq}"
    )

    logging.info(f"Running command for iteration {j}")
    os.system(command)
    end_time = time.time()  # End time of the loop

    logging.info(
        f"Command for iteration {j} completed. Time taken: {end_time - start_time} seconds"
    )

time.sleep(2)
os.system("python plot_comparison3.py")
