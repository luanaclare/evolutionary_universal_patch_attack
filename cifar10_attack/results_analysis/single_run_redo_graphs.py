import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import os
from single_model_redo_graphs import aggregate_to_blocks

def parse_string_list(s):
    return [int(x) for x in s.strip('[]').split(', ')]


if __name__ == "__main__":

    scenarios = ['f1', 'f2','f3']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    for scenario in scenarios:
        main_folder = f'../runs_final/{scenario}'
        # modelNames = ['ResNet50', 'ResNet101', 'VGG16', 'VGG19']
        modelNames = ['ResNet50', 'VGG16']
        gen_change = 20

        for m in range(len(modelNames)):
            modelName = modelNames[m]
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
                        name = r'L$\infty$'
                        avg_name = 'AVG ' + name
                        best_name = "BEST IND'S " + name
                        ps_max = 1

                    matplotlib.rcParams.update({'font.size': 40})

                    data = pd.read_csv(ficheiro_adv_l2)
                    avg_ps = data['perturbation_size_average']
                    std_ps = data['perturbation_size_std']
                    best_ind_ps = data['best_ind_perturbation_size']
                    avg_advq = data['adv_quantity_average']
                    std_advq = data['adv_quantity_std']
                    best_ind_advq = data['best_ind_advq']
                    std_fit = data['fitness_std']

                    ngens = len(avg_ps)
                    gens = list(range(0, ngens))

                    fig, ax = plt.subplots(1, 1)

                    ax.plot(gens[0:-1], avg_ps[0:-1], linestyle='-', label=avg_name)
                    ax.plot(gens[0:-1], best_ind_ps[0:-1], linestyle='-', label=best_name)
                    ax.fill_between(gens[0:-1], avg_ps[0:-1] - std_ps[0:-1], avg_ps[0:-1] + std_ps[0:-1], alpha=0.3)  # Fill between with std

                    for gen in gens:
                        if gen % gen_change == 0:
                            ax.axvline(x=gen, color='gray', linestyle='--')
                    
                    ax.set_xlabel('Generations')
                    ax.set_ylabel(name)
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    ax.set_ylim(0, ps_max + 0.05)
                    ax.set_title(f'{name} across generations')
                    fig.set_size_inches(12, 8)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    plt.savefig(run_folder + f'/{scenario}_{modelName}_ps_' + '.png', bbox_inches='tight')
                    plt.close(fig)

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(gens[0:-1], avg_advq[0:-1], linestyle='-', label="AVG ASR")
                    ax.plot(gens[0:-1], best_ind_advq[0:-1], linestyle='-', label="BEST IND's ASR")
                    ax.fill_between(gens[0:-1], avg_advq [0:-1]- std_advq[0:-1], avg_advq[0:-1] + std_advq[0:-1], alpha=0.3)  # Fill between with std

                    for gen in gens:
                        if gen % gen_change == 0:
                            ax.axvline(x=gen, color='gray', linestyle='--')

                    ax.set_xlabel('Generations')
                    ax.set_ylabel('ASR')
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    ax.set_ylim(0, 1.05)
                    ax.set_title('Avg ASR across generations')
                    fig.set_size_inches(12, 8)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    plt.savefig(run_folder + f'/{scenario}_{modelName}_Adv_ ' + '.png', bbox_inches='tight')
                    plt.close(fig)

                    # FITNESS AND DEPTH
                    fit_data = pd.read_csv(ficheiro_fit)
                    fit_best = fit_data[' fitness generational best']
                    fit_avg = fit_data[' fitness avg']
                    depth_best = fit_data[' depth generational best']
                    depth_avg = fit_data['depth avg']
                    depth_std = fit_data[' depth std']

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(gens[0:-1], fit_avg[0:-1], linestyle='-', label="AVG")
                    ax.plot(gens[0:-1], fit_best[0:-1], linestyle='-', label="BEST")
                    ax.fill_between(gens[0:-1], fit_avg[0:-1] - std_fit[0:-1], fit_avg[0:-1] + std_fit[0:-1], alpha=0.3)  # Fill between with std

                    for gen in gens:
                        if gen % gen_change == 0:
                            ax.axvline(x=gen, color='gray', linestyle='--')

                    ax.set_xlabel('Generations')
                    ax.set_ylabel('Fitness')
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    ax.set_ylim(0, 1.05)
                    ax.set_title('Fitness across generations')
                    fig.set_size_inches(12, 8)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    plt.savefig(run_folder + f'/{scenario}_{modelName}_Fitness_.png', bbox_inches='tight')
                    plt.close(fig)

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(gens[0:-1], depth_avg[0:-1], linestyle='-', label="AVG")
                    ax.plot(gens[0:-1], depth_best[0:-1], linestyle='-', label="BEST")
                    ax.fill_between(gens[0:-1], depth_avg[0:-1] - depth_std[0:-1], depth_avg[0:-1] + depth_std[0:-1], alpha=0.3)  # Fill between with std

                    for gen in gens:
                        if gen % gen_change == 0:
                            ax.axvline(x=gen, color='gray', linestyle='--')

                    ax.set_xlabel('Generations')
                    ax.set_ylabel('Depth')
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    ax.set_ylim(0, 8.05)
                    ax.set_title('Depth across generations')
                    fig.set_size_inches(12, 8)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    plt.savefig(run_folder + f'/{scenario}_{modelName}_Depth_.png', bbox_inches='tight')
                    plt.close(fig)

                    # HEATMAPS AND PATCH AREA
                    patch_folder = f'{run_folder}/logs/perturbation_size_advs_fit_per_gen'
                    gens_heatmap = [0, int( (ngens-1)/2), ngens-1]
                    area_avg = []
                    area_best = []
                    area_std = []
                    for g in range(ngens):
                        ficheiro_pathc = f'{patch_folder}/gen_{g}_perturbation_size_advs_fit.csv' 
                        patch_data = pd.read_csv(ficheiro_pathc)
                        values = patch_data['patch'].values
                        arrays_of_int = [parse_string_list(s) for s in values]
                        arrays_of_int = np.array(arrays_of_int) 
                        w_and_h = arrays_of_int[:, 2:]
                        x_and_y = arrays_of_int[:, :2]

                        if g in gens_heatmap:
                            max_width = 32
                            max_height = 32
                            block_size = 8
                            # Create a grid to store counts
                            heatmap = np.zeros((max_height, max_width))

                            # Count occurrences of each width and height pair
                            for width, height in w_and_h:
                                heatmap[height-1, width-1] += 1  # Subtract 1 to adjust to zero-based indexing

                            # Plot the heatmap
                            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
                            matplotlib.rcParams.update({'font.size': 18})
                            
                            aggregated_heatmap = aggregate_to_blocks(heatmap, block_size)
                            plt.imshow(aggregated_heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=len(w_and_h))
                            plt.colorbar(label='Number of Individuals')
                      #      plt.title('Heatmap of W, H')
                            plt.xlabel('Width', fontsize=18)
                            plt.ylabel('Height', fontsize=18)
                            block_labels = [f'{i}-{i+block_size-1}' for i in range(1, max_width+1, block_size)]
                            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=12)
                            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=12)
                            for i in range(aggregated_heatmap.shape[0]):
                                for j in range(aggregated_heatmap.shape[1]):
                                    plt.text(j, i, f'{int(aggregated_heatmap[i, j])}', ha='center', va='center', color='white', fontsize=8)

                           # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
                            plt.savefig(f'{run_folder}/{scenario}_{modelName}_WH_heatmap_{g}.png', bbox_inches='tight')
                            plt.close()

                            # Create a grid to store counts
                            heatmap = np.zeros((max_height, max_width))

                            # Count occurrences of each width and height pair
                            for x, y in x_and_y:
                                heatmap[y, x] += 1  # Subtract 1 to adjust to zero-based indexing

                            # Plot the heatmap
                            plt.figure(figsize=(10, 8))  # Set figure size (width, height)
                            matplotlib.rcParams.update({'font.size': 18})
                            aggregated_heatmap = aggregate_to_blocks(heatmap, block_size)

                            plt.imshow(aggregated_heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=len(x_and_y))
                            plt.colorbar(label='Number of Individuals')
                            plt.title('Heatmap of X, Y')
                            plt.xlabel('X', fontsize=18)
                            plt.ylabel('Y', fontsize=18)
                            block_labels = [f'{i}-{i+block_size-1}' for i in range(0, max_width, block_size)]
                            plt.xticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=12)
                            plt.yticks(np.arange(len(block_labels)), block_labels, rotation=45, fontsize=12)
                            for i in range(aggregated_heatmap.shape[0]):
                                for j in range(aggregated_heatmap.shape[1]):
                                    plt.text(j, i, f'{int(aggregated_heatmap[i, j])}', ha='center', va='center', color='white', fontsize=8)

                           # plt.gca().invert_yaxis()  # Invert y-axis to match typical image representation
                            plt.savefig(f'{run_folder}/{scenario}_{modelName}_XY_heatmap_{g}.png', bbox_inches='tight')
                            plt.close()

                        areas = w_and_h[:, 0] * w_and_h[:, 1]
                        area_avg.append(np.mean(areas))
                        area_std.append(np.std(areas))
                        area_best.append(w_and_h[0][0]*w_and_h[0][1])

                    matplotlib.rcParams.update({'font.size': 25})

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(gens[0:-1], area_avg[0:-1], linestyle='-', label="AVG Area")
                    ax.plot(gens[0:-1], area_best[0:-1], linestyle='-', label="BEST IND's Area")
                    ax.fill_between(gens[0:-1], np.array(area_avg[0:-1]) - np.array(area_std[0:-1]), np.array(area_avg[0:-1]) + np.array(area_std[0:-1]), alpha=0.3)  # Fill between with std

                    for gen in gens:
                        if gen % gen_change == 0:
                            ax.axvline(x=gen, color='gray', linestyle='--')

                    ax.set_xlabel('Generations')
                    ax.set_ylabel('Area')
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    ax.set_ylim(0, 1025)
                    ax.set_title('Patch area across generations')
                    fig.set_size_inches(12, 8)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
                    plt.savefig(run_folder + f'/{scenario}_{modelName}_Patch_area_.png', bbox_inches='tight')
                    plt.close(fig)

                    # CONFIDENCE IN TRUE LABEL
                    # number_of_ind = len(w_and_h)
                    # best_cy_adv = []
                    # avg_cy_adv = []
                    # std_cy_adv = []

                    # best_cy_no_adv = []
                    # avg_cy_no_adv = []
                    # std_cy_no_adv = []

                    # for g in range(ngens):
                    #     folder_confidence_y = f'{run_folder}/tensor_attack/generation_{g}'
                    #     # best
                    #     ind_means = []
                    #     for i in range(number_of_ind):
                    #         ficheiro_i = f'{folder_confidence_y}/{i}.csv'
                    #         cy_data = pd.read_csv(ficheiro_i)
                    #         # cy_adv = cy_data['adv']
                    #         cy_act_y = cy_data['act_true_label']
                    #         cy_act_y_adv = []

                    #         #if any(cy_adv):             cy_act_y_adv.append(cy_act_y[cy_adv].values)
                    #         cy_act_y_adv.append(cy_act_y.values)
                    #         # else:
                    #         #    cy_act_y_adv = [1]
                            
                    #         ind_mean = np.mean(cy_act_y_adv)
                    #         ind_means.append(ind_mean)
                    #         if i == 0: best_cy_adv.append(np.mean(cy_act_y_adv))

                    #     avg_cy_adv.append(np.mean(ind_means))
                    #     std_cy_adv.append(np.std(ind_means))

                    # fig, ax = plt.subplots(1, 1)
                    # ax.plot(gens[0:-1], avg_cy_adv[0:-1], linestyle='-', label="AVG CY")
                    # ax.plot(gens[0:-1], best_cy_adv[0:-1], linestyle='-', label="BEST IND's CY")
                    # ax.fill_between(gens[0:-1], np.array(avg_cy_adv[0:-1]) - np.array(std_cy_adv[0:-1]), np.array(avg_cy_adv[0:-1]) + np.array(std_cy_adv[0:-1]), alpha=0.3)  # Fill between with std

                    # for gen in gens:
                    #     if gen % gen_change == 0:
                    #         ax.axvline(x=gen, color='gray', linestyle='--')

                    # ax.set_xlabel('Generations')
                    # ax.set_ylabel('CY')
                    # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))  # Ensure integers on x-axis
                    # ax.set_ylim(0, 1.05)
                    # ax.set_title('Confidence in true label across generations')
                    # fig.set_size_inches(12, 8)
                    # plt.legend(loc='lower right')
                    # plt.savefig(run_folder + '/CY_.png')
                    # plt.close(fig)
        

