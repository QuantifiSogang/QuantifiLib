import numpy as np

def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts

def nested_parts(num_atoms, num_threads, upper_triangle=False):
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triangle:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts

def expand_call(kargs):
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out