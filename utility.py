import contextlib
import os
import sys
import time
import traceback

import numpy as np
from numba import njit

print_tmp = print


def print(*args, **kwargs):
    info = traceback.format_stack()[-2]
    end = info.index(",", info.index(",") + 1)
    line_number = traceback.format_stack()[-2][7:end]
    print_tmp(*args, f"{line_number}", **kwargs)


class StopExecution(Exception):
    def _render_traceback_(self):
        return []


def exit():
    raise StopExecution


class HiddenPrints:
    def __enter__(self, enable=True):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        # sys.stdout = open(os.devnull, "w")
        # call the method in question
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            value = func(*args, **kwargs)
        # enable all printing to the console
        # sys.stdout = sys.stdout
        # pass the return value of the method back
        return value

    return func_wrapper


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@njit
def npindex(array, item, start=0):
    for idx, val in np.ndenumerate(array[start:]):
        if val == item:
            return idx


def index(array, condition, start=0):
    for idx, val in enumerate(array[start:]):
        if condition(val):
            return idx


class Timer:
    def __init__(self):
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"Time taken: {self.end - self.start}")
        return self.end - self.start


class NestedDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return self.setdefault(key, NestedDict())


def column(matrix, i):
    return [row[i] for row in matrix]
