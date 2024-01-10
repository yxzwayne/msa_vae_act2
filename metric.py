import os

sVAE_gen_fn = "gen_sVAE_1048576.seqs"
potts_gen_fn = "../seq/gen_potts.seqs"

indep_gen_fn = "../seq/gen2.indep.seqs"
target_fn = "../seq/test.seqs"

model_name = "sVAE"

for i in range(2, 14):
    j = str(i)
    command = (
        "python tee_naive_PI.py "
        + j
        + f" 500 {target_fn} {potts_gen_fn} {indep_gen_fn}"
    )
    os.system(command)

os.system(f"python plot_comparison3.py PI.h5")
