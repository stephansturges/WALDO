import onnx
import os


# Preprocessing: load the ONNX model
#model_path = os.path.join("ONNX_models", "20230420_12class_960_1.onnx")
onnx_model = onnx.load("./ONNX_models/20230420_12class_960_1.onnx")

print("The model is:\n{}".format(onnx_model))

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)
else:
    print("The model is valid!")