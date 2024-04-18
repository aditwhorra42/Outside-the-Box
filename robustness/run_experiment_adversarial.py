from run.experiment_helper import *
from monitoring import *
from utils import *
from data import *
from trainers import *
import os
import pickle
import tensorflow_core as tf
from copy import deepcopy
import json
import uuid
from robustness.constants import (
    PerturbationType,
    ExampleType,
    PerturbationInfo,
    AdversarialExample,
    GeneratedExamples,
)
from typing import List

clustering_threshold = 0.07


def get_FGSM_perturbation(model, data_spec, num_known_classes, eps=50):
    cc_loss = tf.keras.losses.CategoricalCrossentropy()

    images = tf.cast(data_spec._x, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)
        loss = cc_loss(data_spec._y[:num_known_classes], pred)

    gradient = tape.gradient(loss, images)
    perturbation = tf.sign(gradient)

    adv_images = images + (eps / 255.0) * perturbation
    adv_images = tf.clip_by_value(adv_images, -1, 1)

    return adv_images


def get_norm_bounded_perturbation(model, data_spec, eps=0.1):
    # Generate a random perturbation
    perturbation = np.random.randn(*data_spec._x.shape)

    # Normalize the perturbation to have the specified Euclidean norm
    perturbation_norm = np.linalg.norm(
        perturbation.reshape(perturbation.shape[0], -1), 2, axis=-1, keepdims=True
    )
    perturbation_norm = perturbation_norm.reshape(-1, 1, 1, 1)
    normalized_perturbation = perturbation / perturbation_norm * eps

    # Add the perturbation to the original image
    perturbed_images = data_spec._x + normalized_perturbation
    perturbed_images = np.clip(perturbed_images, -1, 1)

    return perturbed_images


def run_data_spec_and_get_predictions(
    model, data_spec, remove_misclassifications: bool = False
):
    data = deepcopy(data_spec)
    values = model.predict(data.x())
    classes = to_classifications(values)
    confidence = []

    incorrect_prediction_indices = []
    for i, (p, gt) in enumerate(zip(classes, data.ground_truths())):
        confidence.append(values[i, gt])
        if p != gt:
            incorrect_prediction_indices.append(i)

    if remove_misclassifications:
        filter_indices = [
            index
            for index in range(len(data._y))
            if index not in incorrect_prediction_indices
        ]
        data.filter(filter_indices)

    return data, confidence, incorrect_prediction_indices


def generate_adversarial_examples_for_known(
    model,
    known_data_spec,
    max_iterations,
    monitor_manager,
    perturbation_info: PerturbationInfo,
    adv_path: str,
    num_known_classes: int,
):
    adversarial_examples = []

    print(f"Initial known dataset size: {known_data_spec._x.shape[0]}")

    # Performing the initial filtering by removing misclassifications
    original_data, _, _ = run_data_spec_and_get_predictions(
        model=model, data_spec=known_data_spec, remove_misclassifications=True
    )

    noisy_data = deepcopy(original_data)

    print(
        f"Size of data after removing initial misclassifications: {noisy_data._x.shape[0]}"
    )

    statistics = Statistics()

    indices_to_ignore = []
    original_image_path = os.path.join(adv_path, "original")
    adv_image_path = os.path.join(adv_path, "adversarial")

    if not os.path.exists(original_image_path):
        os.makedirs(original_image_path)
        os.makedirs(adv_image_path)

    total_adversarial = 0
    misclassified_indices = []
    num_misclassifications_for_iteration = []
    num_adversarial_for_iteration = []

    for iter in range(max_iterations):
        print(f"Performing iteration {iter + 1} for adversarial perturbation.")

        # Getting perturbed images
        if perturbation_info.perturbation_type == PerturbationType.NORM_BOUNDED:
            perturbed_images = get_norm_bounded_perturbation(
                model=model,
                data_spec=noisy_data,
                eps=perturbation_info.eps,
            )
        elif perturbation_info.perturbation_type == PerturbationType.FGSM:
            perturbed_images = get_FGSM_perturbation(
                model=model,
                data_spec=noisy_data,
                eps=perturbation_info.eps,
                num_known_classes=num_known_classes,
            )

        # Create a spec for perturbed images and save in filtered_data so that it can be iteratively used to add perturbations
        noisy_data = DataSpec(x=perturbed_images, y=noisy_data._y)

        _, confidences, incorrect_prediction_indices = (
            run_data_spec_and_get_predictions(model=model, data_spec=noisy_data)
        )

        misclassified_indices.extend(incorrect_prediction_indices)
        misclassified_indices = list(set(misclassified_indices))

        print(
            f"{len(misclassified_indices)}/{len(noisy_data._y)} misclassified after iteration {iter + 1}"
        )

        # Need to now check where the monitor fails and but model gives correct prediction (not in incorrect_prediction_indices)
        history_run = monitor_manager.run(
            model=model, data=noisy_data, statistics=statistics
        )

        acceptances = [
            result.accepts(confidence_threshold=0.5)
            for result in history_run.monitor2results[1]
        ]

        indices_to_ignore.extend(incorrect_prediction_indices)

        monitor_failure_indices = [
            index
            for index, result in enumerate(acceptances)
            if (not result) and (index not in indices_to_ignore)
        ]

        print(
            f"Created {len(monitor_failure_indices)} adversarial examples in {iter + 1}"
        )

        total_adversarial += len(monitor_failure_indices)
        num_adversarial_for_iteration.append(total_adversarial)
        num_misclassifications_for_iteration.append(len(misclassified_indices))

        for f_index in monitor_failure_indices:
            image_name = f"{uuid.uuid4()}.jpg"
            perturbed_image = noisy_data._x[f_index][:, :, 0]
            original_image = original_data._x[f_index][:, :, 0]
            image_distance = np.linalg.norm(original_image - perturbed_image)
            perturbed_save_path = os.path.join(adv_image_path, image_name)
            original_save_path = os.path.join(original_image_path, image_name)
            plt.imsave(perturbed_save_path, perturbed_image, cmap="gray")
            plt.imsave(original_save_path, original_image, cmap="gray")
            adversarial_examples.append(
                AdversarialExample(
                    pertubed_image_path=perturbed_save_path,
                    original_image_path=original_save_path,
                    label=categorical2number(noisy_data._y[f_index]),
                    example_type=ExampleType.KNOWN,
                    num_iterations=iter + 1,
                    model_confidnce=confidences[f_index],
                    monitor_confidences=history_run.monitor2results[1][
                        f_index
                    ]._confidences,
                    image_distance=image_distance,
                )
            )

        # Removing the data that has already created adversarial examples from the next iteration
        indices_to_ignore.extend(monitor_failure_indices)

    return (
        adversarial_examples,
        num_misclassifications_for_iteration,
        num_adversarial_for_iteration,
    )


def generate_adversarial_examples(
    classes_for_experiment: List[int],
    epsilons: List[int],
    monitor_save_base: str,
    adversarial_examples_save_base_path: str,
    max_iterations: int,
):
    model_name, data_name, stored_network_name, total_classes = (
        "MNIST",
        "MNIST",
        "MNIST",
        10,
    )

    for n_classes in classes_for_experiment:
        for eps in epsilons:
            experiment_name = f"norm_bounded_eps_{eps}"

            perturbation_info = PerturbationInfo(
                perturbation_type=PerturbationType.NORM_BOUNDED,
                eps=eps,
            )

            monitor_save_path = os.path.join(
                monitor_save_base, f"MNIST_0-{n_classes-1}.obj"
            )

            adversarial_examples_class_base = os.path.join(
                adversarial_examples_save_base_path,
                f"MNIST_0-{n_classes-1}",
                experiment_name,
            )
            adversarial_examples_class_images = os.path.join(
                adversarial_examples_class_base, "images"
            )
            if not os.path.exists(adversarial_examples_class_images):
                os.makedirs(adversarial_examples_class_images)

            adversarial_examples_json_path = os.path.join(
                adversarial_examples_class_base, "info.json"
            )

            model_statistics = Statistics()

            with open(monitor_save_path, "rb") as infile:
                monitor = pickle.load(infile)

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

            model, history_model = get_model(
                model_name=model_name,
                data_train=data_train_model,
                data_test=data_test_model,
                n_classes=len(labels_network),
                model_trainer=None,
                n_epochs=None,
                batch_size=None,
                statistics=model_statistics,
                model_path=model_path,
            )

            monitor_manager = MonitorManager(
                monitors=[monitor],
                clustering_threshold=clustering_threshold,
                skip_confidence=False,
            )

            layers = set()
            for monitor in monitor_manager._monitors:
                layers.update(monitor.layers())
            monitor_manager._layers = list(layers)

            # Split the data run data spec into known and unknown data specs

            ground_truths = data_run.ground_truths()

            known_indices = []
            for data_index, gt in enumerate(ground_truths):
                if gt in all_classes_network:
                    known_indices.append(data_index)

            unknown_indices = [
                i for i in range(len(ground_truths)) if i not in known_indices
            ]

            known_x, known_y, unknown_x, unknown_y = (
                np.take(data_run._x, known_indices, axis=-0),
                np.take(data_run._y, known_indices, axis=0),
                np.take(data_run._x, unknown_indices, axis=0),
                np.take(data_run._y, unknown_indices, axis=0),
            )

            known_data_spec = DataSpec(
                x=known_x, y=known_y, classes=all_classes_network
            )
            unknown_data_spec = DataSpec(
                x=unknown_x,
                y=unknown_y,
                classes=[c for c in all_classes_rest if c not in all_classes_network],
            )

            (
                known_adv_examples,
                num_misclassifications_for_iteration,
                num_adversarial_for_iteration,
            ) = generate_adversarial_examples_for_known(
                model=model,
                known_data_spec=known_data_spec,
                max_iterations=max_iterations,
                monitor_manager=monitor_manager,
                perturbation_info=perturbation_info,
                adv_path=adversarial_examples_class_images,
                num_known_classes=len(all_classes_network),
            )

            if len(known_adv_examples) > 0:
                min_dist = min(
                    [adv_example.image_distance for adv_example in known_adv_examples]
                )
                norm_min_dist = min_dist / (perturbation_info.eps * max_iterations)
            else:
                min_dist = None
                norm_min_dist = 1.0

            adv_ratio = len(known_adv_examples) / known_data_spec._x.shape[0]
            norm_adv_ratio = 1 - adv_ratio

            generated_examples = GeneratedExamples(
                examples=known_adv_examples,
                perturbation_info=perturbation_info,
                min_dist=min_dist,
                norm_min_dist=norm_min_dist,
                adv_ratio=adv_ratio,
                norm_adv_ratio=norm_adv_ratio,
                num_misclassifications_for_iteration=num_misclassifications_for_iteration,
                num_adversarial_for_iteration=num_adversarial_for_iteration,
            )

            with open(adversarial_examples_json_path, "w") as file:
                json.dump(generated_examples.json(), file)


if __name__ == "__main__":
    classes_for_experiment = [2, 5, 7]
    epsilons = [1, 3, 5, 7, 10]
    monitor_save_base = "<path-to-trained-monitors>"
    adversarial_examples_save_base_path = "<path-to-save-results>"
    max_iterations = 10

    generate_adversarial_examples(
        classes_for_experiment=classes_for_experiment,
        epsilons=epsilons,
        monitor_save_base=monitor_save_base,
        adversarial_examples_save_base_path=adversarial_examples_save_base_path,
        max_iterations=max_iterations,
    )
