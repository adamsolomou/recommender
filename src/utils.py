from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import math


def get_batches(iterable, batch_size=64, do_shuffle=True):
    if do_shuffle:
        iterable = shuffle(iterable)

    length = len(iterable)
    for ndx in range(0, length, batch_size):
        iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
        yield iterable_batch


def root_mean_square_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
