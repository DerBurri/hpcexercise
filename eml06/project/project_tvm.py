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
from model import SRCNN


# import matplotlib for image
import matplotlib.pyplot as plt


model = SRCNN(num_channels=1)
# load the weights from pth file and convert to cpu
state_dict = model.state_dict()
for n, p in torch.load('/home/mburr/tvm/hpcexercise-1/eml06/project/data/srcnn_x4.pth', map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        model.state_dict()[n].copy_(p)
    else: 
        raise KeyError(n)
    

image_org = Image.open('/home/mburr/tvm/hpcexercise-1/eml06/project/image_downsample/images/baby_downsized_4x.jpg.bmp').convert('RGB')

# resize the image to 1920 x1080
image = image_org.resize((853, 1280), resample=Image.BICUBIC)
image_ycbcr = image.convert('YCbCr')
#extract the Y channel
image_y, image_cb, image_cr = image_ycbcr.split()
y = np.array(image_y).astype(np.float32)
y /= 255.
y = np.expand_dims(y, axis=0)  # Add batch dimension
y = np.expand_dims(y, axis=0)  # Add channel dimension

# We grab the TorchScripted model via tracing
dummy_input = torch.from_numpy(y)
scripted_model = torch.jit.trace(model, torch.from_numpy(y)).eval()
scripted_model.save("srcnn.pt")
# Preprocess the image and convert to tensor
from torchvision import transforms


#my_preprocess = transforms.Compose(
#    [
#        transforms.Resize(256),
#        transforms.CenterCrop(224),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#    ]
#)
#
#image = my_preprocess(image)
# Convert the TorchScript model to Relay IR
input_shape = dummy_input.shape
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


# We grab the TorchScripted model via tracing

# create relay ir from torch


#torch.onnx.export(model, dummy_input, "srcnn.onnx", input_names=["input"], output_names=["output"])

# Load the ONNX model
#onnx_model = onnx.load("srcnn.onnx")

# Convert the ONNX model to Relay IR
#shape_dict = {"input": (1, 1, 224, 224)}
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)


from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
module = partition_for_arm_compute_lib(mod, params)



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

# # Parameter Tune the model
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


# # Load and preprocess the image
# img_url= "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
# img_name = "cat.png"
# img_path = download_testdata(img_url, img_name, module='data')
# img = Image.open(img_path).resize((1920, 1080)).convert('L')
# img.show()
# input_image = np.array(img).astype("float32")
# input_image = np.expand_dims(input_image, axis=(0,1))
module.set_input("input0", y)

plt.imshow(y.squeeze())
plt.savefig('input.png')
# Run the model
module.run()

# Get output
out = module.get_output(0)
# calculate the psnr with skleran
#from skimage.metrics import peak_signal_noise_ratio
#psnr = peak_signal_noise_ratio(image.squeeze(), out.squeeze())
#print psnr
#print("PSNR: {:.2f}".format(psnr))
print("Output shape:", out.shape)

# save output image
preds = out.numpy().squeeze(0).squeeze(0)
#merge the output with the cb and cr channels
output = np.array([preds, image_cb, image_cr]).transpose([1, 2, 0])
output = Image.fromarray(np.uint8(output), mode='YCbCr').convert('RGB')

output.save("output.png")


ftimer = module.module.time_evaluator("run", dev, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))