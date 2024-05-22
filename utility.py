import random


def shuffle_and_split_data(data, ratio, seed=0):
    """
    Shuffle the data list with a given seed and split it according to the given ratio.

    Parameters:
    - data: original list of data.
    - ratio: The ratio of the data to select after shuffling (0 < ratio <= 1).
    - seed: The random seed for reproducibility.

    Returns:
    - A tuple of two lists: the selected subset of data and the remaining data.
    """
    
    random.seed(seed)
    data_copy = data[:]
    
    random.shuffle(data_copy)
    
    split_index = int(len(data_copy) * (1-ratio))
    
    selected_data = data_copy[:split_index]
    remaining_data = data_copy[split_index:]
    
    return selected_data, remaining_data




