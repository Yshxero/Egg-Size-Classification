import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "egg_model.pkl")

with open(MODEL_PATH, "rb") as f:
    obj = pickle.load(f)

print("Top-level type:", type(obj))

# If it's a dict, show keys + types
if isinstance(obj, dict):
    print("Dict keys:", list(obj.keys()))
    for k, v in obj.items():
        print(f"{k}: {type(v)}")
else:
    # If it's a sklearn pipeline/model, print its class name
    print("Object repr:", obj)
    # sklearn Pipeline has steps
    if hasattr(obj, "steps"):
        print("Pipeline steps:")
        for name, step in obj.steps:
            print(" -", name, type(step))
