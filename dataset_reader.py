
import sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def read_dataset():
    rounds_data=pd.read_csv('exported_rounds.csv')
    rounds_data.head()
    return rounds_data.as_matrix()

def read_opp_moves():
    moves_data=pd.read_csv('exported_moves.csv')
    moves_data.head()
    return moves_data.as_matrix()[:,0]