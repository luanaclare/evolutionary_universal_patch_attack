import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import os

def parse_string_list(s):
    return [int(x) for x in s.strip('[]').split(', ')]

def aggregate_to_blocks(heatmap, block_size):
    max_height, max_width = heatmap.shape
    block_height = max_height // block_size
    block_width = max_width // block_size
    
    aggregated_heatmap = np.zeros((block_height, block_width))
    
    for i in range(block_height):
        for j in range(block_width):
            block = heatmap[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            aggregated_heatmap[i, j] = np.sum(block)
    
    return aggregated_heatmap

def plot_metric(gens, data_dict, type, metric_name, y_label, ylim, file_name):
    fig, ax = plt.subplots(1, 1)                                                         
    i = 0
    for scenario, data in data_dict.items():
        avg_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        ax.plot(gens[0:-1], avg_data[0:-1], linestyle='-', label=f"{targets[i]} {type} {metric_name}")
        ax.fill_between(gens[0:-1], avg_data[0:-1] - std_data[0:-1], avg_data[0:-1] + std_data[0:-1], alpha=0.3)  # Fill between with std
        i = i + 1
    
    for gen in gens:
        if gen % gen_change == 0:
            ax.axvline(x=gen, color='gray', linestyle='--')

    ax.set_xlabel('Generations')
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
    ax.set_ylim(0, ylim)
    ax.set_title(f'{metric_name} across generations')
    fig.set_size_inches(12, 8)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    if not os.path.exists('graphs'):
        os.mkdir('graphs')
        
    plt.savefig(f'graphs/{file_name}', bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":

    scenarios = ['f1', 'f2', 'f3']
    targets = ['3%', '5%', '10%']
    # modelNames = ['ResNet50','ResNet101', 'VGG16', 'VGG19']
    modelNames = ['ResNet50', 'VGG16']
    gen_change = 20

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    for modelName in modelNames:
        all_avg_pss = {scenario: [] for scenario in scenarios}
        all_best_pss = {scenario: [] for scenario in scenarios}
        all_avg_asr_treinos = {scenario: [] for scenario in scenarios}
        all_best_asr_treinos = {scenario: [] for scenario in scenarios}
        all_avg_fitnesses = {scenario: [] for scenario in scenarios}
        all_best_fitnesses = {scenario: [] for scenario in scenarios}
        all_avg_depths = {scenario: [] for scenario in scenarios}
        all_best_depths = {scenario: [] for scenario in scenarios}
        all_avg_areas = {scenario: [] for scenario in scenarios}
        all_best_areas = {scenario: [] for scenario in scenarios}
        #all_avg_confidences = {scenario: [] for scenario in scenarios}
        #all_best_confidences = {scenario: [] for scenario in scenarios}

        for scenario in scenarios:
            main_folder = f'../runs_final/{scenario}'
            for folder_name in os.listdir(main_folder):
                if folder_name.startswith(modelName):
                    run_folder = os.path.join(main_folder, folder_name)
                    ficheiro_adv_l2 = f'{run_folder}/gen_perturbation_size_advs_fit.csv'
                    ficheiro_fit = f'{run_folder}/evolution_{folder_name}.csv'

                    # PERTURBATION SIZE AND ADVERSARIAL SUCCESS RATE
                    l2 = False
                    if l2:
                        name = 'L2'
                        avg_name = 'AVG ' + name
                        best_name = "BEST IND'S " + name
                        ps_max = 32 * np.sqrt(3)
                    else:
                        name = r"L$\infty$"
                        avg_name = 'AVG ' + name
                        best_name = "BEST IND'S " + name
                        ps_max = 1

                    matplotlib.rcParams.update({'font.size': 40})

                    data = pd.read_csv(ficheiro_adv_l2)
                    avg_ps = data['perturbation_size_average']
                    best_ind_ps = data['best_ind_perturbation_size']

                    avg_advq = data['adv_quantity_average']
                    best_ind_advq = data['best_ind_advq']

                    fit_data = pd.read_csv(ficheiro_fit)
                    fit_best = fit_data[' fitness generational best']
                    fit_avg = fit_data[' fitness avg']
                    depth_best = fit_data[' depth generational best']
                    depth_avg = fit_data['depth avg']

                    all_best_asr_treinos[scenario].append(best_ind_advq)
                    all_avg_asr_treinos[scenario].append(avg_advq)

                    all_best_pss[scenario].append(best_ind_ps)
                    all_avg_pss[scenario].append(avg_ps)

                    all_best_fitnesses[scenario].append(fit_best)
                    all_avg_fitnesses[scenario].append(fit_avg)

                    all_best_depths[scenario].append(depth_best)
                    all_avg_depths[scenario].append(depth_avg)

                    patch_folder = f'{run_folder}/logs/perturbation_size_advs_fit_per_gen'
                    ngens = len(avg_ps)
                    gens_heatmap = [0, int((ngens-1)/2), ngens-1]
                    area_avg = []
                    area_best = []

                    best_cy_adv = []
                    avg_cy_adv = []

                    for g in range(ngens):
                        ficheiro_pathc = f'{patch_folder}/gen_{g}_perturbation_size_advs_fit.csv'
                        patch_data = pd.read_csv(ficheiro_pathc)
                        values = patch_data['patch'].values
                        arrays_of_int = [parse_string_list(s) for s in values]
                        arrays_of_int = np.array(arrays_of_int)
                        w_and_h = arrays_of_int[:, 2:]
                        x_and_y = arrays_of_int[:, :2]
                        number_of_ind = len(w_and_h)

                        # folder_confidence_y = f'{run_folder}/tensor_attack/generation_{g}'
                        # ind_means = []
                        # for i in range(number_of_ind):
                        #     ficheiro_i = f'{folder_confidence_y}/{i}.csv'
                        #     cy_data = pd.read_csv(ficheiro_i)
                        #     cy_act_y = cy_data['act_true_label']
                        #     cy_act_y_adv = []

                        #     cy_act_y_adv.append(cy_act_y.values)

                        #     ind_mean = np.mean(cy_act_y_adv)
                        #     ind_means.append(ind_mean)
                        #     if i == 0:
                        #         best_cy_adv.append(np.mean(cy_act_y_adv))

                        # avg_cy_adv.append(np.mean(ind_means))

                        areas = w_and_h[:, 0] * w_and_h[:, 1]
                        area_avg.append(np.mean(areas))
                        area_best.append(w_and_h[0][0] * w_and_h[0][1])

                    #all_best_confidences[scenario].append(best_cy_adv)
                    #all_avg_confidences[scenario].append(avg_cy_adv)
                    all_avg_areas[scenario].append(area_avg)
                    all_best_areas[scenario].append(area_best)

        ngens = len(avg_ps)
        gens = list(range(0, ngens))

        fold = f'images_final'
        if not os.path.exists(fold):
            os.mkdir(fold)
        

        plot_metric(gens, all_avg_pss, 'AVG', "L$\infty$", name, ps_max + 0.05,
                    f'{fold}/{modelName}_Perturbation_SizeAVG.png')
        plot_metric(gens, all_best_pss, 'BEST', "L$\infty$", name, ps_max + 0.05,
                    f'{fold}/{modelName}_Perturbation_SizeBEST.png')
        
        plot_metric(gens, all_avg_asr_treinos, 'AVG', "ASR", "ASR", 1.05,
                    f'{fold}/{modelName}_Adv_ASRAVG.png')
        plot_metric(gens, all_best_asr_treinos, 'BEST', "ASR", "ASR", 1.05,
                    f'{fold}/{modelName}_Adv_ASRBEST.png')
        
        plot_metric(gens, all_avg_fitnesses, 'AVG', "Fitness", "Fitness", 1.05,
                    f'{fold}/{modelName}_FitnessAVG.png')
        plot_metric(gens, all_best_fitnesses, 'BEST', "Fitness", "Fitness", 1.05,
                    f'{fold}/{modelName}_FitnessBEST.png')
        
        plot_metric(gens, all_avg_depths, 'AVG', "Depth", "Depth", 8.05,
                    f'{fold}/{modelName}_DepthAVG.png')
        plot_metric(gens, all_best_depths, 'BEST', "Depth", "Depth", 8.05,
                    f'{fold}/{modelName}_DepthBEST.png')
        
        plot_metric(gens, all_avg_areas, 'AVG', "Patch Area", "Area", 1025,
                    f'{fold}/{modelName}_Patch_AreaAVG.png')
        plot_metric(gens, all_best_areas, 'BEST', "Patch Area", "Area", 1025,
                    f'{fold}/{modelName}_Patch_AreaBEST.png')
        
       # plot_metric(gens, all_avg_confidences, 'AVG', "Confidence in True Label", "CY", 1.05,
       #             f'{fold}/{modelName}_ConfidenceAVG.png')
       # plot_metric(gens, all_best_confidences, 'BEST', "Confidence in True Label", "CY", 1.05,
       #             f'{fold}/{modelName}_ConfidenceBEST.png')
     
