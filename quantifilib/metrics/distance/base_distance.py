import numpy as np
import pandas as pd

class BaseDistance() :
    def __init__(self, data:pd.DataFrame, verbose:bool = True) :
        self.data = data
        self.verbose = verbose