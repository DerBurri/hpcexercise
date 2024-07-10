from tvm.autotvm.tuner import XGBTuner
import tvm
from tvm import autotvm
from tvm import relay
import onnx
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata
import torch
from torch import nn
import numpy as np
from PIL import Image

# import matplotlib for image
import matplotlib.pyplot as plt



class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SRCNN(num_channels=1)
dummy_input = torch.randn(1, 1, 224, 224)
# load the weights from pth file and convert to cpu
model.load_state_dict(torch.load('/home/mburr/tvm/hpcexercise-1/eml06/project/srcnn_x3.pth', map_location=torch.device('cpu')))





torch.onnx.export(model, dummy_input, "srcnn.onnx", input_names=["input"], output_names=["output"])

# Load the ONNX model
onnx_model = onnx.load("srcnn.onnx")

# Convert the ONNX model to Relay IR
shape_dict = {"input": (1, 1, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)



target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Extract tasks
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Define tuning options
tuning_option = {
    'log_filename': 'srcnn_tuning.log',
    'tuner': 'xgb',
    'n_trial': 10,
    'early_stopping': 60,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=10)
    ),
}

# Parameter Tune the model
# for task in tasks:
#      tuner = XGBTuner(task)
#      tuner.tune(n_trial=tuning_option['n_trial'],
#                 early_stopping=tuning_option['early_stopping'],
#                 measure_option=tuning_option['measure_option'],
#                 callbacks=[autotvm.callback.log_to_file(tuning_option['log_filename'])])
    
with autotvm.apply_history_best(tuning_option['log_filename']):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)


# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("srcnn.tar")
lib.export_library(lib_fname)

# Obtain an RPC session from the remote device.
local_demo = False

if local_demo:
    remote = rpc.LocalSession()
else:
    # Update the host and po    port = 9090
    host = "10.100.100.131"
    port = 9090 
    remote = rpc.connect(host, port)

# Upload the library to the remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("srcnn.tar")

# Create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))

# Set input data
#input_data = np.random.rand(1, 1, 224, 224).astype("float32")


# Load and preprocess the image
img_url= "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module='data')
img = Image.open(img_path).resize((224, 224)).convert('YCbCr')
img.show()
input_image = np.array(img).astype("float32")
input_image = np.expand_dims(input_image, axis=(0,1))
module.set_input("input", tvm.nd.array(input_image))

plt.imshow(input_image.squeeze(), cmap='ycbcr')
plt.savefig('input.png')
# Run the model
module.run()

# Get output
out = module.get_output(0).numpy()
print("Output shape:", out.shape)

# save output image
plt.imshow(out.squeeze(), cmap='ycbcr')
plt.savefig('output.png')

ftimer = module.module.time_evaluator("run", dev, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))