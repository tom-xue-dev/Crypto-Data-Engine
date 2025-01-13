import pickle

with open("data_signal.pkl", "rb") as f:
    loaded_data = pickle.load(f)

for t,group in loaded_data.groupby("time"):
    print(group)