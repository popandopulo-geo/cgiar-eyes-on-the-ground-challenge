import contextlib
import joblib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given askimage.transform"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def estimate_maximum_batch_size(model, device, input_shape):
    model = model.to(device)
    batch = 2
    
    while True:
    
        x = torch.ones((batch,) + input_shape).to(torch.float32).to(device)
        torch.cuda.synchronize()
        try:
            model(x)
        except torch.cuda.OutOfMemoryError:
            batch = batch // 2
            break
        else:
            batch *= 2

    return batch

def dict2str(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict2str(value)
        else:
            d[key] = str(value)

    return d

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def vector2prediction(predictions):
    # Find the indices of the last maximum value along the second dimension (axis=1)
    max_indices = torch.argmax(torch.flip(predictions, dims=[1]), dim=1)
    max_indices = predictions.size(1) - max_indices
    
    # Calculate the mask for rows where the sum is greater than 0
    mask = (predictions.sum(dim=1) > 0).to(torch.int64)
    
    # Compute the result using tensor operations
    return max_indices * mask

def get_intervals(input_tensor, cutpoints):
    # Sort the cutpoints and input tensor
    sorted_cutpoints = torch.sort(cutpoints).values
    sorted_input = torch.sort(input_tensor).values

    # Find the indices where the input_tensor crosses the cutpoints
    interval_indices = torch.searchsorted(sorted_cutpoints, sorted_input)

    return interval_indices

def get_confusion_matrix(targets, predictions, n_classes):
    confusion_matrix = torch.zeros(n_classes, n_classes, device=predictions.device, dtype=predictions.dtype)

    indices = targets * n_classes + predictions
    ones = torch.ones_like(indices, device=predictions.device, dtype=predictions.dtype)
    confusion_matrix.view(-1).scatter_add_(0, indices, ones)

    return confusion_matrix

def draw_confusion_matrix(confusion_matrix, eps=1e-6):
    fig, axes = plt.subplots(1,2, figsize=(20,10))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    n_confusion_matrix = confusion_matrix / (confusion_matrix.sum(axis=1)[..., None] + eps)
    disp = ConfusionMatrixDisplay(confusion_matrix=n_confusion_matrix)
    disp.plot(cmap="Blues", ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix")

    plt.close(fig)
    return fig

def get_classes_priora_proba(split, meta):
    N = split.shape[0]

    _, volumes = torch.unique(torch.tensor(meta.loc[split.index].extent), return_counts=True)
    return volumes / N

def get_costs_proportional(n_classes, distance):
    indices = torch.arange(n_classes).to(torch.float32)
    return distance(indices.unsqueeze(1), indices)