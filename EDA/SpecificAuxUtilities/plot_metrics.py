import numpy as np
from pathlib import Path

Base_path = Path('../outputs/MAIN/')
for f in range(1):
    for cv in range(3):
        c = np.load(str(Base_path.joinpath(str(f+1)).joinpath(str(cv+1)).joinpath('FScore.npy')))