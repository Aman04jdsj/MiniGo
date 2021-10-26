import pandas as pd

gamma = 0.95

df = pd.read_csv("black.csv", header=None)
mul = 1
for index, row in df.iloc[::-1].iterrows():
    if int(row[1]) != 0 and row[1] != "draw":
        mul = gamma * float(row[1])
    elif int(row[1]) == 0:
        df.at[index, 1] = mul
        mul *= gamma
df.to_csv("black.csv", header=None, index=None)
df = pd.read_csv("white.csv", header=None)
mul = 1
for index, row in df.iloc[::-1].iterrows():
    if int(row[1]) != 0 and row[1] != "draw":
        mul = gamma * float(row[1])
    elif int(row[1]) == 0:
        df.at[index, 1] = mul
        mul *= gamma
df.to_csv("white.csv", header=None, index=None)
