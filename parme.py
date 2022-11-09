import os
import time
import pickle
import logging
import calendar

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import grb_vars_to_ndarray, grb_vars_shape


# Global Gurobi setting
current_GMT = time.gmtime()
timestamp = calendar.timegm(current_GMT)
log_file = "gurobi.ParMe.{}.log".format(timestamp)

