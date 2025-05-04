import pickle

with open("network_final-gaussian-n2c.pickle", "rb") as f:
    obj = pickle.load(f)

print(type(obj))