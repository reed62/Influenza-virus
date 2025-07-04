import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

training_sizes = [100, 500, 1000, 3000, 5000, 10000]
r2_scores = []
r2_errors = []
r2_scores_all = []
results_summary = []
summary_save_path = "../results/cv_summary.csv"
train_file = "../data/wt_filter_gate_train.csv"
model = "deepcnnlstm_in"
virus_name = "in"
date = "0702"
k = 5

for training_size in training_sizes:
    result = subprocess.run(
    f"python train.py {model} cv -k {k} --virus-name={virus_name} --date={date} "
    f"--num-train={training_size} --input-file={train_file} --save",
    shell=True
)


    result_csv_path =f"../validation/{model}_cv_{virus_name}_{date}_{k}fold_{training_size}.csv"
    df = pd.read_csv(result_csv_path)

    r2_list = df["validation_pearsonr"].tolist()
    for i, r in enumerate(r2_list):
        results_summary.append({
            "training_size": training_size,
            "run": i,
            "r": r
        })
        r2_scores_all.append(r)

    average_r2 = np.mean(r2_list)
    std_err = np.std(r2_list) / np.sqrt(len(r2_list))

    r2_scores.append(average_r2)
    r2_errors.append(std_err)

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(summary_save_path, index=False)


plt.figure(figsize=(10, 6))
bar_width = 0.4
index = np.arange(len(training_sizes))

plt.bar(index, r2_scores, bar_width, alpha=0.8, color='lightgrey',
        label='Average R2 Score', yerr=r2_errors, capsize=7)


index_all = np.repeat(index, k)
plt.scatter(index_all, r2_scores_all, color='red', zorder=5, alpha=0.5, label='Individual R2 Scores')

plt.xlabel('Training Size', fontsize=15)
plt.ylabel('Pearson r', fontsize=15)
plt.xticks(index, training_sizes, fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.rcParams['svg.fonttype']= 'none'
plt.savefig("../fig/training_size.svg", format='svg', bbox_inches='tight')
plt.show()
