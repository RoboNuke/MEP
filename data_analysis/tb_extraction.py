import pandas as pd
import requests
from csv import reader
import os
import sys

def getURL(tag, fold, exp_mode, state_mode, trial):
    return f'http://localhost:8080/data/plugin/scalars/scalars?tag={tag}&run={fold}/{exp_mode}/{state_mode}_{trial}&format=csv'

def tb_data(fold, state_mode, num_trials):
    tags = [
            "charts/episodic_length", 
            "charts/episodic_return",
            "charts/eval_episodic_length",
            "charts/eval_episodic_return",
            "charts/eval_success_rate",
            "charts/success_rate"
          ]
    
    exp_modes = ["baseline", "MEP"]

    fdf = None
    for exp_mode in exp_modes:
        for trial in range(num_trials):
            for tag in tags:
                r = requests.get(getURL(tag, fold, exp_mode, state_mode, trial+1))
                data = r.text
                data_csv = reader(data.splitlines())
                data_csv = list(data_csv)
                df = pd.DataFrame(data_csv)
                headers = df.iloc[0]
                df = pd.DataFrame(df.values[1:], columns=headers)
                df['exp_mode'] = exp_mode
                df['data_tag'] = tag
                df['trial'] = trial
                if fdf is None:
                    fdf = df
                else:
                    fdf = pd.concat([fdf,df], ignore_index=True)

    return fdf

if __name__=="__main__":
    if len(sys.argv) > 4:
        fold = sys.argv[1]
        state_mode = sys.argv[2]
        trials = int(sys.argv[3])
        save_path = str(sys.argv[4])
        tb_data(fold, state_mode, trials).to_csv(save_path + "collected_data.csv", index=False)
    else:
        tb_data("test_PushCube-v1_2024-07-13_16:56", "state", 5).to_csv('runs/test_PushCube-v1_2024-07-13_16:56/collected_data.csv', index=False)

    print("Data Extracted and Collected")