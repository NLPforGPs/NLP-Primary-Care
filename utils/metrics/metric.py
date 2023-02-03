import numpy as np
from sklearn.metrics import f1_score, classification_report, roc_auc_score, precision_score, recall_score


def test_random_baseline(X_test, y_test):
    f = 0
    p = 0
    r = 0

    N = X_test.shape[0]
    nclasses = y_test.shape[1]

    nsamples = 1000
    for sample in range(nsamples):
        # random_labels = np.random.rand(y_hot.shape[0], y_hot.shape[1])
        # random_labels = random_labels > (counts / np.sum(counts))[None, :]
        random_labels = np.random.randint(0, 2, (N, nclasses))
        p += precision_score(random_labels, y_test, average='macro')
        r += recall_score(random_labels, y_test, average='macro')
        f += f1_score(random_labels, y_test, average='macro')

    print(f'Precision = {p / nsamples}')
    print(f'Recall = {r / nsamples}')
    print(f'F1 score = {f / nsamples}')

    return p, r, f


def evaluate_per_class(targets,
                       predictions):
    """
    Evaluates the discrete classification labels.
    :param targets: A matrix where each row is a one-hot representation of the gold labels for the sample.
    :param predictions: A matrix where each row is a one-hot representation of the predicted labels for the sample.
    :return:
    """

    keepclasses = np.any(targets, axis=0)
    targets = targets[:, keepclasses]
    predictions = predictions[:, keepclasses]

    return f1_score(targets, predictions, average=None, zero_division=0), \
           precision_score(targets, predictions, average=None, zero_division=0), \
           recall_score(targets, predictions, average=None, zero_division=0)


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

    return f1_score(targets, predictions, average='macro', zero_division=0), \
           precision_score(targets, predictions, average='macro', zero_division=0), \
           recall_score(targets, predictions, average='macro', zero_division=0)


def evaluate_probabilities(targets, predictions):
    """
    Evaluates the probabilities output by a classifier using the ROC curve.
    :param targets: A matrix where each row is a one-hot representation of the gold labels for the sample.
    :param predictions: A matrix where each row is a one-hot representation of the predicted labels for the sample.
    :return:
    """

    keepclasses = np.any(targets, axis=0)
    targets = targets[:, keepclasses]
    predictions = predictions[:, keepclasses]

    auroc = roc_auc_score(targets, predictions, average='macro')  # [:, keepclasses], predictions[:, keepclasses])
    # print(f'area under ROC curve: {auroc}')

    return auroc


def error_analysis(predictions, probabilities, split_nums, segments, transcript_ids, save_file):
    '''
    This method is used to save the predictions for error analysis
    predictions: nested list for each segment, [[label1, label2],..]
    probabilities: probability for each segment [exmaples, classes]
    split_nums: the number of segments for each transcript
    segments: nested list, how many segments in each transcript, [[segment1, segment2],..]
    transcript_ids: list, transcript id, [transcript1, transcript2,...]
    '''

    probs = np.max(probabilities, axis=1)

    # flatten all segments
    segments = [seg for segment in segments for seg in segment]
    # align segment ids and transcript ids
    segment_ids = []
    seg_transcript_ids = []
    for i, num in enumerate(split_nums):
        segment_ids.extend(list(range(num)))
        seg_transcript_ids.extend([transcript_ids[i]] * num)
    assert len(segment_ids) == len(predictions)
    with open(save_file, 'w') as f:
        f.write('transcript_id\tsegment_id\ttext\tlabel\tprobability\n')
        for i, label in enumerate(predictions):
            f.write(seg_transcript_ids[i] + '\t' + str(segment_ids[i]) + '\t' + segments[i] + '\t' + ",".join(
                label) + '\t' + str(probs[i]) + '\n')
