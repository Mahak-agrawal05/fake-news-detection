#read data from fake and true csv

import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print(fake.head())
print()
print(true.head())
print()
print(fake["title"].head())
print()
print(fake.shape)
print()
print(fake.columns)
print()