# -*- coding: utf-8 -*-
# @Time    : 2024/3/10
# @Author  : quanchenliu
# @Function: 多项式回归

import math
import torch
import numpy as np
from d2l import torch as d2l
from torch import nn


array = [[1, 2, 3]]                             # [1, 3]
features = [[1, ], [2, ], [3, ]]                # [3,1]
poly_features = np.power(features, array)
print(poly_features)