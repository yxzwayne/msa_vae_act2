import sys

fn = "deepersequence/seq/6M.exper.seqs"

with open(fn, "r") as file:
    lines = file.readlines()[30000:40000]

new_file_name = "deepersequence/seq/6M.exper_10k.seqs"
with open(new_file_name, "w") as new_file:
    new_file.writelines(lines)
