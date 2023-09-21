import torch
import onnxruntime as rt
import os
import warnings

warnings.filterwarnings("ignore")

for model in os.listdir("./"):
    if model.endswith(".onnx"):
        try:
            providers = ['CUDAExecutionProvider']
            sess = rt.InferenceSession("./" + model, providers=providers)
            print("------------------------------------------------")
            print(f"Loaded model OK : {model}")
        except:
            print(f"Error loading model: {model}")
            break
