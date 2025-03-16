import pandas as pd
import numpy as np
from ast import literal_eval

data = pd.read_csv('processed_file.csv')

def parse_or_none(x):
    try:
        return np.array(literal_eval(x))
    except Exception:
        print("Exception!!")
        return None

data['embedding_vector'] = data['embedding_vector'].apply(parse_or_none)

# Drop rows where parsing failed
data = data.dropna(subset=['embedding_vector']).reset_index(drop=True)
print(len(data))
