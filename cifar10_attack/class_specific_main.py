import sys
import os
import tensorflow as tf
import numpy as np
import csv 
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset, DataLoader

absolute_path = os.path.abspath('../')
sys.path.append(absolute_path)

from tensorgp.engine import *

import os
from vgg import VGG 
from resnet import ResNet50, ResNet101

tf.get_logger().setLevel('ERROR')

def create_engine(resolution):
    # resolution = [resolution, resolution, 3] #jncor could be better #[4096, 4096,3] # 4096x4096
    #fset = {'abs', 'add', 'and', 'cos', 'div', 'exp', 'frac', 'if', 'log',
     #   'max', 'mdist', 'min', 'mult', 'neg', 'or', 'pow', 'sin', 'sqrt',  'sub', 'tan', 'xor','lerp', 'sstepp', 'sign', 'clip', 'mod', 'len'}

    # setup Engine for coin
    new_engine = Engine(target_dims=resolution,
                    codomain=[-1, 1],
                    domain=[-1, 1],
                    effective_dims=2,
                    do_final_transform=True,
                    final_transform=[-255, 255],
                    tf_type=np.int8,
                    # polar_coordinates=True,
                    # do_polar_mask=True,
                    # polar_mask_value="scalar(-1.0, -1.0, -1.0)",
                    device=dev,
                    mkdirrun=False)
    
    return new_engine, resolution


def get_patch(individual):
    w = individual['w']
    h = individual['h']
    tree = individual['tree']
    # exp = tree.get_str()
    new_engine, re = create_engine([w,h,3])
    # _, tr = str_to_tree(exp, engine.get_terminal_set().set)
    tensor = new_engine.domain_mapping(tree.get_tensor(new_engine))
    tensor_cpu = tensor.cpu()
    tensor_cpu = np.array(tensor_cpu)

    patches = []
    bound_x = [0, 32 - w]
    bound_y = [0, 32 - h]

    xs  = np.random.randint(bound_x[0], bound_x[1] + 1, size=5)
    ys  = np.random.randint(bound_y[0], bound_y[1] + 1, size=5)

    for i in range(5):
        patch = np.zeros((32,32,3))
        patch[xs[i]:xs[i]+w, ys[i]:ys[i]+h] = tensor_cpu[0:w, 0:h]
        patches.append(patch)

    return xs, ys, patches

def compute_l2_distance(original_images, perturbed_images):
    # Compute the L2 distance between two images
    l2_distances = []
    for img1, img2 in zip(original_images, perturbed_images):
        subtraction = img1/255 - img2/255
        l2_distance = np.linalg.norm(subtraction)
        l2_distances.append(l2_distance)
    return l2_distances
    #subtraction = original_image - perturbed_image
    #l2_distance = np.linalg.norm(subtraction, axis=(1,2))

def compute_linf_distance(original_image, perturbed_image):
    # Compute the L∞ distance between two images
    linf_distances = []
    for img1, img2 in zip(original_image, perturbed_image):
        linf_distance = np.max(np.abs(img1/255 - img2/255))
        linf_distances.append(linf_distance)
    return linf_distances

def apply_perturbation(image, perturbation_tensor):
    perturbed_image = image + perturbation_tensor
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 255)
    return perturbed_image

def transform_image(image, mean, std):
    epsilon = 1e-8  # Small value to avoid division by zero
    std = std + epsilon

    transform_test2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2])),
    ])
     
    transformed_image = transform_test2(image)
    return transformed_image

def fitness_function(**kwargs):
    # read parameters    
    population = kwargs.get('population')
    generation = kwargs.get('generation')
    scenario = kwargs.get('scenario')
   # print("Generation: ", generation)
    tensors = kwargs.get('tensors')
    f_path = kwargs.get('f_path')
    objective = kwargs.get('objective')
    _resolution = kwargs.get('resolution')

    mini_subset_number = int(generation/generational_change)
    if generation == number_generations:
        mini_subset_number = mini_subset_number - 1
        mini_subset_indices = np.reshape(evo_indices, -1)
        mini_subset_indices = list(mini_subset_indices)
    else:
        mini_subset_number = (generation // generational_change) % len(evo_indices)
        mini_subset_indices = evo_indices[mini_subset_number]

    mini_subset_size = len(mini_subset_indices)
    best_ind = 0

    # set objective function according to min/max
    fit = 0
    condition = lambda: (fit > max_fit) # maximizing
    max_fit = float('-inf')

    fitnesses = []
    pertsizes = []
    advs = []
    number_tensors = len(tensors)
    targets_array = np.array(testset.targets)
    true_labels = targets_array[mini_subset_indices]

    if 'f1' in scenario: target = 0.03
    elif 'f2' in scenario: target = 0.05
    elif 'f3' in scenario: target = 0.1
    elif 'f4' in scenario: target = 10/255

    dataset = testset.data[mini_subset_indices]


    with tf.device('/gpu:0'):

        index_tensor = 0

        for tensor in tensors:

            # Map the function to the dataset to perturb each image
            xs, ys, patches = get_patch(population[index_tensor])

            best_patch_fitness = float('-inf')
            for p in range(5):
                patch = patches[p]
                if tf.reduce_all(tf.equal(patch, 0)):
                    patch_fitness = 0
                    adv_quantity = 0
                    avg_perturbation = 0
                    perturbed_images = testset_transformed.data[mini_subset_indices]
                    predicted_numbers = targets_array[mini_subset_indices]
                    wrong_classification = []
                    activations = np.zeros((mini_subset_size,))
                    confidences_y = np.zeros((mini_subset_size,))

                else:
                    perturbed_images = np.array([apply_perturbation(image, patch) for image in dataset])
                   # perturbed_images = torch.stack([apply_perturbation(img, patch_cuda) for img in dataset])
                    true_labels = torch.tensor(true_labels, dtype=torch.long)

                    mean = np.mean(perturbed_images, axis=(0, 1, 2))   # Normalize to [0, 1]
                    #mean = torch.mean(perturbed_images, dim=(0, 1, 2))
                    std = np.std(perturbed_images, axis=(0, 1, 2))  
                    #std = torch.std(perturbed_images, dim=(0,1,2))
                    perturbed_images_transformed = torch.stack([transform_image(image,mean,std) for image in perturbed_images])
                    perturbed_images_tensor = torch.tensor(perturbed_images_transformed)
                    perturbed_transformed_dataset = TensorDataset(perturbed_images_tensor, true_labels)
                    testloader_perturbed = DataLoader(perturbed_transformed_dataset, batch_size=mini_subset_size, shuffle=False, num_workers=0)

                    all_predictions = []
                    with torch.no_grad():
                        for scaled_perturbed_images, true_labels in testloader_perturbed:
                            scaled_perturbed_images = scaled_perturbed_images.to('cuda')
                            scaled_perturbed_images = scaled_perturbed_images.float()
                            predictions = classif(scaled_perturbed_images)
                            all_predictions.append(predictions)

                    predictions = torch.cat(all_predictions).cpu().numpy()
                    max_logits = predictions.max(axis=1, keepdims=True)
                    shifted_logits = predictions - max_logits
                    exp_logits = np.exp(shifted_logits)
                    predictions = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    predicted_numbers = predictions.argmax(axis=1)
                    activations = predictions.max(axis=1)

                    wrong_classification = (predicted_numbers != true_labels.cpu().numpy()).nonzero()[0]

                    success = [1 if i in wrong_classification else 0 for i in range(mini_subset_size)]
                    adv_quantity = len(wrong_classification)/mini_subset_size

                    confidences_y = [predictions[i][true_labels[i].item()] for i in range(mini_subset_size)]
                    confidence_component = [1/(confidences_y[i] + 1) for i in range(mini_subset_size)] # range: 0.5 - 1
                    confidence_component = [(confidence_component[i] - 0.5)/(1 - 0.5) for i in range(mini_subset_size)]
                    

                    # perturbation
                    # l2
                    # perturbation_sizes = compute_l2_distance(dataset, perturbed_images)
                    
                    # linf
                    
                    perturbation_sizes = compute_linf_distance(dataset, perturbed_images)
                    avg_perturbation = np.mean(perturbation_sizes)
                    perturbation_component = [1/(perturbation_size +1) for perturbation_size in perturbation_sizes] 

                    # l2
                    # perturbation_component = [(perturb - 1/(32*np.sqrt(3) + 1))/(1 - 1/(32*np.sqrt(3) + 1)) for perturb in perturbation_component]
                    # linf
                    perturbation_component = [(perturb - 0.5)/(1 - 0.5) for perturb in perturbation_component]
                    
                    w1 = 0.2
                    w2 = 0.5
                    w3 = 0.3

                    if avg_perturbation > target:
                        scores = [ w1*perturbation_component[i] for i in range(mini_subset_size)]

                    else:
                        scores = [ w1 + w2*success[i] + w3*confidence_component[i] for i in range(mini_subset_size)]

                    patch_fitness = np.mean(scores) # +  1 / (np.std(scores) + 1)

                if patch_fitness > best_patch_fitness:
                    best_patch = patch
                    best_patch_x = xs[p]
                    best_patch_y = ys[p]
                    best_patch_fitness = patch_fitness
                    best_patch_adv_quantity = adv_quantity
                    best_patch_avg_perturbation = avg_perturbation
                    best_patch_perturbed_images = perturbed_images                    
                    best_patch_predicted_numbers = predicted_numbers
                    best_patch_wrong_classification = wrong_classification
                    best_patch_activations = activations
                    best_patch_act_true_label = confidences_y

            fitnesses.append(best_patch_fitness)
            advs.append(best_patch_adv_quantity)
            pertsizes.append(best_patch_avg_perturbation)

            population[index_tensor]['perturbation'] = best_patch
            population[index_tensor]['x'] = best_patch_x
            population[index_tensor]['y'] = best_patch_y
            population[index_tensor]['perturbed_images'] = best_patch_perturbed_images
            population[index_tensor]['predicted_numbers'] = best_patch_predicted_numbers
            population[index_tensor]['indexes'] = mini_subset_indices
            population[index_tensor]['labels'] = targets_array[mini_subset_indices]
            population[index_tensor]['activations'] = np.array(best_patch_activations)
            population[index_tensor]['adv_list'] = best_patch_wrong_classification
            population[index_tensor]['act_true_label'] = best_patch_act_true_label

            index_tensor += 1

        
        for index in range(number_tensors):
            fit = fitnesses[index]

            if condition():
                max_fit = fit
                best_ind = index

            population[index]['fitness'] = fit
            population[index]['perturbation_size'] = pertsizes[index]
            population[index]['adv_q'] = advs[index]

    return population, best_ind



if __name__ == "__main__":
   
    resolution = [1, 1, 3]

    # GP params
    dev = 'cuda'  # device to run, write '/cpu_0' to tun on cpu
    

    number_generations = 100
    pop_size = 200
    tour_size = 4
    mut_prob_tree = 0.35
    mut_prob_wh = 0.05
    cross_prob = 1
    max_tree_dep = 8
    edims = 2

    # new params
    total_subset_size = 100 # 1000
    mini_subset_size = 20 # 200
    generational_change = 20 # 25
    qnt_subsets = int(total_subset_size/mini_subset_size)
    per_class = int(total_subset_size/10)

    # Load data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)
    
    testset_transformed = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Load models 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #modelNames = ['ResNet101', 'ResNet50']
    #cls = [ResNet101(), ResNet50()]
    modelNames = ['ResNet50']
    cls = [ResNet50()]

    class_specifics = [0, 1, 2,3,4,5,6,7,8,9]

    for class_specific in class_specifics:
        # scenarios = [f'class_{class_specific}_f1', f'class_{class_specific}_f3']
        scenarios = [f'f2_class_{class_specific}']

        for sce in scenarios:
            for m in range(len(modelNames)):
                classif = cls[m]
                modelName = modelNames[m]
                checkpoint_file = f'../pytorch-cifar-master/checkpoint/{modelName}/ckpt.pth' # '../pytorch-cifar-master/checkpoint/ResNet/ckpt.pth']
                classif = classif.to(device)

                if device == 'cuda':
                    classif = torch.nn.DataParallel(classif)

                checkpoint = torch.load(checkpoint_file)
                classif.load_state_dict(checkpoint['net'])
                classif.eval()

                selected_images_file = pd.read_csv(f'selected_images_EvoSet/{modelName}_cifar_attack_subset_2020_{5}x{200}.csv')
                filtered_images_file = selected_images_file[selected_images_file['label'] == class_specific]
                random_indices = filtered_images_file['index']

                part_size = len(random_indices) // qnt_subsets
                evo_indices = [random_indices[i * part_size:(i + 1) * part_size] for i in range(qnt_subsets)]

                seeds = [654, 114, 25, 759, 281, 250, 228, 142, 754, 104, 692, 758, 913, 558, 89, 604, 432, 32, 30, 95, 223, 238, 517, 616, 27, 574, 203, 733, 665, 718]

                for r in range(1, 15):
                    seed = seeds[r]
                    np.random.seed(seed)
                    print('RUN', r)
                    # Engine
                    engine = Engine(fitness_func=fitness_function,
                                    population_size=pop_size,
                                    tournament_size=tour_size,
                                    mutation_rate=mut_prob_tree,
                                    mutation_rate_wh= mut_prob_wh,
                                    crossover_rate=cross_prob,
                                    max_tree_depth = max_tree_dep,
                                    target_dims=resolution,
                                    method='ramped half-and-half',
                                    objective='maximizing',
                                    device=dev,
                                    stop_criteria='generation',
                                    codomain = [-1, 1],
                                    domain=[-1, 1],
                                    do_final_transform = True,
                                    final_transform = [-255, 255],
                                    bloat_control='off',
                                    tf_type=np.int8,
                                    stop_value=number_generations,
                                    effective_dims = edims,
                                    seed = seed,
                                    debug=0,
                                    save_to_file=1, 
                                    save_graphics=True,
                                    show_graphics=False,
                                    read_init_pop_from_file=None,
                                    generational_change=generational_change,
                                    modelName=modelName,
                                    scenario=sce,
                                    )

                    engine.run()
