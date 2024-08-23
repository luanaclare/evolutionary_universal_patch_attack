import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 20})

# Define the function to plot the data
def plot_asr_data(scenarios, targets, asr_type):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'pink']

    asr_label_map = {
        'ASR Evolution': 'ASR Evolution',
        'ASR Test': 'ASR Test',
        'ASR Overall': 'ASR Overall'
    }

    for scenario in scenarios:
        main_folder = f'../runs_final/{scenario}'
        data = pd.read_csv(f'{main_folder}/dataset.csv')
        
        models = data['Model'].unique()

        for i, model in enumerate(models):
            model_data = data[data['Model'] == model]
            asr_values = model_data[asr_type].values
            color = colors[i % len(colors)]
            
            if scenarios.index(scenario) == 0:
                plt.scatter(targets[scenarios.index(scenario)], asr_values, label=f'{model}', color=color)
            else: 
                plt.scatter(targets[scenarios.index(scenario)], asr_values, color=color)

    plt.title(f'{asr_label_map[asr_type]} Values Across Scenarios')
    plt.xlabel(r"L$\infty$")
    plt.ylabel('ASR')
    plt.xticks(targets, [f'{t:.2f}' for t in targets])    
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    if not os.path.exists('graphs'):
        os.mkdir('graphs')
        
    plt.savefig(f'graphs/{asr_type.replace(" ", "_")}_xPS.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    scenarios = ['f1', 'f2', 'f3']
    targets = [0.03, 0.05, 0.1]

    plot_asr_data(scenarios, targets, 'ASR Evolution')
    plot_asr_data(scenarios, targets, 'ASR Test')
    # plot_asr_data(scenarios, targets, name, 'ASR Overall')



