import concurrent
import os
import numpy as np
import concurrent.futures


def evaluate_all_elements(func, arr, args):
    """
    Evaluates all elements of a NumPy array using a specified function in parallel processes.

    Parameters:
    func (function): The function to apply to each element.
    arr (numpy.ndarray): The input NumPy array.
    args (tuple): Additional arguments to pass to the function.

    Returns:
    numpy.ndarray: An array containing the results of applying the function to all elements.
    """
    num_cores = os.cpu_count()

    if len(arr) < num_cores:
        num_cores = len(arr)

    split_arr = np.array_split(arr, num_cores)
    args_list = [(func, batch,) + args for batch in split_arr]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(evaluate_batch, args_list))

    return np.concatenate(results)


def evaluate_batch(args):
    """
    Evaluates a batch of elements using a specified function.

    Parameters:
    args (tuple): A tuple containing the function, batch array, and additional function arguments.

    Returns:
    numpy.ndarray: An array containing the results of applying the function to the batch.
    """
    func, batch, *func_args = args
    return np.apply_along_axis(func1d=func, axis=1, arr=batch, args=func_args)

