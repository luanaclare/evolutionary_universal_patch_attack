# Evolutionary Universal Patch Attack

This repository contains the code for the Evolutionary Universal Patch Attack and Evolutionary Class Specific Patch Attack presented in the work "Evolutionary Black-Box Adversarial Attacks" by Luana Clare.

Target models are from https://github.com/kuangliu/pytorch-cifar .

Pre-attack Files
* analyse_class_accuracy.py → calculates class accuracy
* classify_images.py → separate correctly classified images
* select_images_for_EvoSet → select images for EvoSet

Evolution Files
* main.py → run universal attack
* class_specific_main.py → run class specific attack

Attack Files
* attack_with_expression.py → attack TestSet with random individual or inputed expression
* transfer_attack.py → after one attack created for original model, attack other models with it
* attack_TestSet → attack images in TestSet
* class_specific_attack_TestSet → attack images in TestSet, class specific version

Analysis Files
* analyse_avg_runs_results.py → get average results from all runs
* asr_ps_graph.py → create a graph of Adversarial Success Rate vs Perturbation Size for models
* class_speficic_analyse_avg_runs_results.py → get average results from all runs for class specific attack
* class_specific_find_best_seed.py → find best seed
* coverage_heatmaps.py → do coverage heatmaps (avg between runs)
* create_individual_channels.py → generate image with individual separated in rgb channels
* create_tree_image.py → create image with tree from json tree string
* find_best_seed.py → find best seed
* generate_class_specific_test_grid.py → generate grid with adversarial examples from TestSet
* get_best_run_results → get results from best run
* single_run_redo_graphs.py → create graphs for individual runs
* single_model_redo_graphs.py → create graphs for a model in a scenario, averaging the runs for that model in the scenario
* single_model_one_img_redo_graphs.py → create one image per model with different scenarios.
* transfer_attack_analyse_results.py → get average results from all runs in transfer attacks
* str_to_json.py → string to json string for tree
 
