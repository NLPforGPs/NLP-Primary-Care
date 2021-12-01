def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    if not os.path.exists(dir):
        os.makedirs(dir)

    filepath = os.path.join(dir, '%s.pt' % name)
    torch.save(state, filepath)


def softmax(logits):
    normalized = np.exp(logits - np.max(logits, axis = -1, keepdims=True))
    return normalized / np.sum(normalized, axis=-1, keepdims=True)

def merge_predictions(record_cnt, predictions):
        cum_sum = np.cumsum(record_cnt)
        assert cum_sum[-1] == len(predictions)
        final_predictions = np.zeros((len(record_cnt), predictions.shape[1]))
        prev = 0
        for i in range(len(cum_sum)):
            entry = cum_sum[i]
            final_predictions[i] = np.any(predictions[prev:entry], axis=0).astype(int)
            prev = entry
        return final_predictions