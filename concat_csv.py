import sys
import pandas as pd

out = sys.argv[1]
dfs = [pd.read_csv(p) for p in sys.argv[2:]]
pd.concat(dfs, ignore_index=True).to_csv(out, index=False)