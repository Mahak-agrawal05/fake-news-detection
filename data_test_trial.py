import pandas as pd
Data = {
    "Name" : ["Mahak", "Rahuk", "Aman"],
    "Marks" : [90, 82, 95]
}
df = pd.DataFrame(Data)
print(df)
print(df["Name"])
print(df["Marks"].mean())