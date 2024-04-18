from run.experiment_helper import *
from monitoring import *
from utils import *
from data import *
from trainers import *

def train_and_save_monitor():
    seed = 0
    classes_for_experiment = [2, 5, 7]
    monitor_save_base = "/home/aditwhorra_nl/Outside-the-Box/trained_monitors"

    # instance options
    model_name, data_name, stored_network_name, total_classes = (
        "MNIST",
        "MNIST",
        "MNIST",
        10,
    )

    for n_classes in classes_for_experiment:
        monitor_save_path = os.path.join(
            monitor_save_base, f"MNIST_0-{n_classes-1}.obj"
        )

        # load instance
        (
            data_train_model,
            data_test_model,
            data_train_monitor,
            data_test_monitor,
            data_run,
            model_path,
            classes_string,
        ) = load_instance(n_classes, total_classes, stored_network_name)

        # create monitor
        monitors = [box_abstraction_MNIST()]
        monitor_manager = MonitorManager(
            monitors, clustering_threshold=clustering_threshold, skip_confidence=False
        )

        set_random_seed(seed)

        # construct statistics wrapper
        statistics = Statistics()

        # load data
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

        # load network model or create and train it
        model, history_model = get_model(
            model_name=model_name,
            data_train=data_train_model,
            data_test=data_test_model,
            n_classes=len(labels_network),
            model_trainer=None,
            n_epochs=None,
            batch_size=None,
            statistics=statistics,
            model_path=model_path,
        )

        print(
            (
                "Data: classes {} with {:d} inputs (monitor training), classes {} with {:d} inputs (monitor test), "
                + "classes {} with {:d} inputs (monitor run)"
            ).format(
                classes2string(data_train_monitor.classes),
                data_train_monitor.n,
                classes2string(data_test_monitor.classes),
                data_test_monitor.n,
                classes2string(data_run.classes),
                data_run.n,
            )
        )

        # normalize and initialize monitors
        monitor_manager.normalize_and_initialize(model, len(labels_rest))

        # train monitors
        monitor_manager.train(
            model=model,
            data_train=data_train_monitor,
            data_test=data_test_monitor,
            statistics=statistics,
        )

        monitor = monitor_manager.monitors()[0]
        with open(monitor_save_path, "wb") as outpath:
            pickle.dump(monitor, outpath)
