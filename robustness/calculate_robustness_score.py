from run_experiment_adversarial import AdversarialExample, GeneratedExamples
from experiment_helper import load_instance
from data import get_data_loader
from typing import List
import os
import json
import numpy as np

def robustness_score(eps, adversarial_examples: List[AdversarialExample], test_set_size: int, num_iterations: int):
    perturbed_score = 0
    d_max = num_iterations * eps
    
    score_unperturbed = test_set_size - len(adversarial_examples)
    
    for adv in adversarial_examples:
        perturbed_score += adv.image_distance / d_max

    score = perturbed_score + score_unperturbed
    
    normalised_score = score / test_set_size
    
    return normalised_score
    

if __name__ == "__main__":
    classes_for_experiment = [2, 5, 7]
    epsilon = [1, 3, 5, 7, 10]
    adversarial_save_path = "/home/aditwhorra_nl/Outside-the-Box/final_experiments"
    num_iterations = 10

    # instance options
    model_name, data_name, stored_network_name, total_classes = (
        "MNIST",
        "MNIST",
        "MNIST",
        10,
    )

    for n_classes in classes_for_experiment:
        #TODO: Store test set size in adversarial results and clean this up later.
        (
                data_train_model,
                data_test_model,
                data_train_monitor,
                data_test_monitor,
                data_run,
                model_path,
                classes_string,
            ) = load_instance(n_classes, total_classes, stored_network_name)

        (
            all_classes_network,
            labels_network,
            all_classes_rest,
            labels_rest,
        ) = get_data_loader(data_name)(
                data_train_model=data_train_model,
                data_test_model=data_test_model,
                data_train_monitor=data_train_monitor,
                data_test_monitor=data_test_monitor,
                data_run=data_run,
        )
        
        ground_truths = data_run.ground_truths()

        known_indices = []
        for data_index, gt in enumerate(ground_truths):
            if gt in all_classes_network:
                known_indices.append(data_index)
        
        known_x, known_y = (
            np.take(data_run._x, known_indices, axis=-0),
            np.take(data_run._y, known_indices, axis=0),
        )
        
        test_set_size = known_x.shape[0]
        
        adversarial_examples_class_base = os.path.join(
            adversarial_save_path,
            f"MNIST_0-{n_classes-1}"
        ) 
        for eps in epsilon:
            
            experiment_name = f"norm_bounded_eps_{eps}"
            experiment_path = os.path.join(adversarial_examples_class_base, experiment_name)
            
            with open(os.path.join(experiment_path, "info.json"), "r") as infile:
                loaded_data = json.load(infile)
                if type(loaded_data) == str:
                    loaded_data = json.loads(loaded_data)
        
                generated_examples = GeneratedExamples(**loaded_data)
            
            robustness = robustness_score(eps=eps, adversarial_examples=generated_examples.examples, test_set_size=test_set_size, num_iterations=num_iterations)
            
            print(f"Score for MNIST_0-{n_classes-1} for epsilon {eps} is {robustness}")
        