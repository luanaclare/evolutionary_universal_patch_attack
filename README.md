# Evolutionary Universal Patch Attack

This repository contains the code for the Evolutionary Universal Patch Attack and Evolutionary Class Specific Patch Attack presented in the work "Evolutionary Black-Box Adversarial Attacks" by Luana Clare.

Despite the successful use of Deep Neural Networks in diverse real-world scenarios, recent research has shown that even state-of-the-art networks face robustness issues as they are vulnerable to adversarial examples. Adversarial examples are manipulated inputs generated to subvert neural network's outputs with small alterations in the original input. In this process, the network is the target model. An adversarial example is crafted during an attack, which can happen in white-box scenarios, where the adversary has extensive knowledge of the target model, or in black-box scenarios, where the adversary only has access to the target model's outputs. Image-specific adversarial attacks craft a unique perturbation for each original input and have been thoroughly explored in the literature. Evolutionary computation has risen as a successful strategy in generating adversarial examples in black-box attacks, as the attack can be modeled as an optimization problem. Previous research has shown the existence of Universal Adversarial Perturbations (UAPs), a single perturbation capable of turning multiple original inputs into adversarial examples. This dissertation proposes a novel evolutionary black-box data-driven adversarial attack for the generation of UAPs in non-targeted scenarios, with an adaptation to class-specific scenarios. To obtain the desired perturbation, an adaptation of TensorGP was implemented to use Genetic Programming to evolve functions and two positive integer values - $patch_{w}$ and $patch_{h}$. The phenotype of the individual is acquired by executing the tree and generating an image with a width and height defined by the integer values. This image is the patch, which is added to an original image to perturb it. The position of the patch in the original image is defined by two coordinates - $patch_{x}$ and $patch_{y}$ - that align with the top left corner of the patch. The perturbation size is controlled by a $L_{\infty}$ threshold of 3%, 5% or 10%.

The Evolutionary Universal Patch Attack perturbed 1,000 CIFAR-10 test dataset images, targeting the VGG16 and the ResNet50 networks. The images were all correctly classified by the model under attack before being perturbed.

Target models are from https://github.com/kuangliu/pytorch-cifar .

The files for the attack are organized as follows: 

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
 
