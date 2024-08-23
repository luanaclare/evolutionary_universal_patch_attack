from setup_cifar import CIFAR, CIFARModel
import numpy as np
import pandas as pd
import tensorflow as tf
import sys 
import torchvision

modelName = 'VGG16'

# Load CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)

y_test = testset.targets

# Redirect prints to a text file
with open(f'correctly_classified_images/{modelName}_analyse_dataset_output.txt', 'w') as output_file:
    # Redirect standard output to the file
    sys.stdout = output_file

    print('All:')
    for i in range(10):
        aux = [pred for pred in y_test if pred == i]
        print(i, '-', len(aux))

    print("Correctly classified:")
    images_df = pd.read_csv(f'correctly_classified_images/{modelName}_correctly_classified_images.csv')
    images_idx = images_df['image id']
    images_labels = images_df['true label']

    for i in range(10):
        subset_class = i
        right_class = [idx for idx, label in zip(images_idx, images_labels) if label == subset_class]
        print(i, '-', len(right_class))

    # Reset standard output to the console
    sys.stdout = sys.__stdout__
