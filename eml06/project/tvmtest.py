import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_executor as runtime
from tvm.contrib.download import download_testdata

import torch
import torchvision
from PIL import Image
import numpy as np

# one line to get the model
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)

synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())

    input_name = "input0"
shape_list = [(input_name, x.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
# we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

local_demo = False

if local_demo:
    target = tvm.target.Target("llvm")
else:
    #target = tvm.target.arm_cpu("rasp3b")
    # The above line is a simple form of
    target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2712 -mtriple=armv8-a+fp+simd -mattr=+neon')

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.tar")
lib.export_library(lib_fname)

# obtain an RPC session from remote device.

batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "10.100.100.131"
    port = 9090
    remote = rpc.connect(host, port)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")

# create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))
# set input data
module.set_input(input_name, tvm.nd.array(x.astype("float32")))
# run
module.run()
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(synset[top1]))