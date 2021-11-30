import numpy as np
from sklearn.metrics import f1_score, classification_report


def evaluate_classifications(targets,
                             predictions,
                             class_names,
                             show_report=False):
    """
    Evaluates the discrete classification labels.
    :param targets: A matrix where each row is a one-hot representation of the gold labels for the sample.
    :param predictions: A matrix where each row is a one-hot representation of the predicted labels for the sample.
    :param class_names: The names of the classes in an ordered list.
    :return:
    """

    keepclasses = np.any(targets, axis=0)
    targets = targets[:, keepclasses]
    predictions = predictions[:, keepclasses]
    class_names = np.array(class_names)[keepclasses]

    if show_report:
        # print(targets)
        # print(predictions)
        print(
            f"classification_report:\n{classification_report(targets, predictions, target_names=class_names, zero_division=0)}"
        )

    return f1_score(targets, predictions, average='macro')