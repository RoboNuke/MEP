import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

filepath = "runs/DNH_state_PickCube-v1_2024-07-13_17:44/"

tags = [
        "charts/episodic_length", 
        "charts/episodic_return",
        "charts/eval_episodic_length",
        "charts/eval_episodic_return",
        "charts/eval_success_rate",
        "charts/success_rate"
        ]

names = [
    "Episodic Length",
    "Episodic Return",
    "Evaluation Episodic Length",
    "Evaluation Episodic Return",
    "Evaluation Success Rate",
    "Success Rate"
]

units = [
    "Steps",
    "Summed Reward",
    "Steps",
    "Summed Reward",
    "Percent",
    "Percent"
]

maxes = [
    50,
    -1,
    50,
    -1,
    1.0,
    1.0
]

exp_modes = ["baseline", "MEP"]

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        task_name = sys.argv[2]
        state_space = sys.argv[3]

    df = pd.read_csv(filepath + "collected_data.csv")

    trials = max(df['trial']) + 1

    colors = ['b','r']

    for j, data_tag in enumerate(tags):
        fig, ax = plt.subplots()

        for i, exp_mode in enumerate(exp_modes):
            steps = []
            vals = []

            for trial in range(trials):
                fdf = df[
                    (df['trial'] == trial) & 
                    (df['exp_mode']==exp_mode) & 
                    (df['data_tag']==data_tag)
                ]
                steps.append(fdf['Step'].to_numpy())
                vals.append(fdf['Value'].to_numpy())

            mu, error = tolerant_mean(vals)
            x, _ = tolerant_mean(steps)
            ci = 1.96 * np.std(mu) / np.sqrt(len(mu))

            ax.plot(x, mu, color=colors[i], label=exp_mode)
            if maxes[j] == -1:
                ax.fill_between(x, (mu-ci), (mu + ci), color=colors[i], alpha=.1)
            else:
                print( (mu+ci).shape, maxes[i])
                toppy = np.zeros((mu+ci).shape)
                for k in range((mu+ci).shape[0]):
                    toppy[k] = min( (mu+ci)[k], maxes[j])
                ax.fill_between(x, (mu-ci), toppy, color=colors[i], alpha=.1)
        ax.set_title(task_name + " (" + state_space + ") " + names[j])
        #plt.legend(exp_modes)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel(units[j])
        plt.savefig(filepath + "plots/" + names[j] + ".png")
    #plt.show()

    print("Plots plotted pretty-like")
