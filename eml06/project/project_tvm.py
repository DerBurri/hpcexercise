#TODO Pruning
#TODO PSNE for float32
#TODO PSNR for float16
#TODO PSNR for int8 quantisation
#TODO PLOT int8 weights after quantisation
#TODO Inspection
#TODO Plot Weights after Pruning
#TODO Plot Weights after quanitzation
#TODO RPC Tuining

#TODO Plot Weights before pruning -> finished
import os
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
from tvm.relay.quantize import quantize
import torch
from torch import nn
import numpy as np
from PIL import Image
from model import SRCNN
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)


# import matplotlib for image
import matplotlib.pyplot as plt
# quantize relay network with standard TVM, post-training quantization
def quantize_network( relay_network, params, quantization_config=None, print_nw=False):
    # https://github.com/apache/tvm/blob/main/tests/python/nightly/quantization/test_quantization_accuracy.py
    # https://github.com/apache/tvm/blob/main/tests/python/nightly/quantization/test_quantization_accuracy_for_vit.py
    # error: https://discuss.tvm.apache.org/t/quantization-quantizer-fails-with-specific-layers/6140
    print('quantize model...')
    # use defaults if nothing special is required
    if quantization_config is None:
        quantization_config = relay.quantize.qconfig(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=32,
            dtype_input='int8',
            dtype_weight='int8',
            dtype_activation='float32',
            calibrate_mode='global_scale',
            global_scale=8.0,
            weight_scale="power2",
            skip_dense_layer=False,
            skip_conv_layers=[0],
            do_simulation=False,
            round_for_shift=True,
            debug_enabled_ops=None,
            rounding="UPWARD",
            calibrate_chunk_by=-1,
            partition_conversions="disabled",
            )
    # transformation of network
    with quantization_config: 
        qfunc = relay.quantize.quantize(relay_network, params=params)
    if print_nw:
        print(qfunc)
    print('quantized')
    return qfunc

def prune_weights_between_limits_permanently(model, lower_limit, upper_limit):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("pruned layer: ", name)
            # Create a mask for weights to keep
            mask = (module.weight.data >= lower_limit) & (module.weight.data <= upper_limit)
            # Apply mask to set weights outside the limits to zero instead of removing them
            module.weight.data[~mask] = 0
            # Note: This approach maintains the original shape and structure of the weights
            # remove the weights completely when set to 0 to reduce computatins


def prune_filters_based_on_average_weight(model, lower_limit, upper_limit):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate the average weight for each filter
            filter_avgs = module.weight.data.mean(dim=[1, 2, 3])
            # Create a mask for filters to keep
            filter_mask = (filter_avgs >= lower_limit) & (filter_avgs <= upper_limit)
            # Invert mask to select filters to prune
            filters_to_prune = np.where(filter_mask.numpy() == False)[0]
            # Prune filters
            module = prune_conv_layer(module, filters_to_prune)

def prune_conv_layer(conv, filters_to_prune):
    """Prunes the specified filters from a convolutional layer."""
    # Create a mask to keep filters not in filters_to_prune
    mask = torch.ones(conv.out_channels, dtype=torch.bool)
    mask[filters_to_prune] = False
    # Use the mask to select filters
    conv.weight.data = conv.weight.data[mask, :, :, :]
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[mask]
    # Adjust the number of output channels
    conv.out_channels = len(conv.weight.data)
    return conv



def plot_weights_histogram(model, layer_name, ax):
    layer = model.get_submodule(layer_name)
    weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for histogram

    ax.hist(weights, bins=50, density=True, alpha=0.7, color='skyblue', label=f'{layer_name}')
    ax.set_title(f'Weight Distribution')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.5)

    mu, std = weights.mean(), weights.std()
    ax.text(0.7, 0.9, f'{layer_name}\nMean: {mu:.4f}, Std Dev: {std:.4f}', transform=ax.transAxes, fontsize=9)


# Load model and image
model = SRCNN(num_channels=1)
state_dict = model.state_dict()
for n, p in torch.load('/home/mburr/tvm/hpcexercise-1/eml06/project/data/srcnn_x4.pth', map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        model.state_dict()[n].copy_(p)
    else:
        raise KeyError(n)
    
# prune weights


lower_limit = -0.0001 # for measurement 1
upper_limit = 0.0001 # for measurement 1
#lower_limit = -0.05 # for measurement 2
#upper_limit = 0.05 # for measurement 2
#lower_limit = -0.075 # for measurement 3
#upper_limit = 0.075 # for measurement 3
#lower_limit = -0.1 # for measurement 4
#upper_limit = 0.1 # for measurement 4
#lower_limit = -0.15 # for measurement 5
#upper_limit = 0.15 # for measurement 5


#prune_weights_between_limits_permanently(model, lower_limit, upper_limit)
#prune_filters_based_on_average_weight(model, lower_limit, upper_limit)

# Assuming `model` is already defined and loaded as shown in the excerpt
conv_layers = [name for name, _ in model.named_modules() if 'conv' in name]

# Create a single figure and axis for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Define a list of colors for the layers
colors = ['skyblue', 'salmon', 'gold', 'lightgreen', 'violet']

max_abs_weight = 0

for layer_name, color in zip(conv_layers, colors):
    layer = model.get_submodule(layer_name)
    weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for histogram
    max_abs_weight = max(max_abs_weight, np.abs(weights).max())
    ax.hist(weights, bins=50, density=True, alpha=0.7, color=color, label=f'{layer_name}')

ax.set_xlim(left=-max_abs_weight, right=max_abs_weight)  # Center zero by setting symmetric xlim
ax.set_title('Weight Distribution Across Layers')
ax.set_xlabel('Weight Value')
ax.set_ylabel('Density')
ax.grid(axis='y', alpha=0.5)
ax.legend()


plt.savefig(f'weights.png')
compare_image = Image.open('/home/mburr/tvm/hpcexercise-1/eml06/project/image_downsample/images/baby.jpg').convert('RGB')
# Preprocess Image
image_org = Image.open('/home/mburr/tvm/hpcexercise-1/eml06/project/image_downsample/images/baby_downsized_4x.jpg.bmp').convert('RGB')
image = image_org.resize((853, 1280), resample=Image.BICUBIC)

image_org = np.array(image_org).astype(np.float32)
ycbcr_org = convert_rgb_to_ycbcr(image_org)

image = np.array(image).astype(np.float32)
ycbcr = convert_rgb_to_ycbcr(image)

y = ycbcr[..., 0] / 255.
y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)


# Convert to Relay IR
scripted_model = torch.jit.trace(model, y).eval()
input_shape = y.shape
input_name = "input0"
shape_list = [(input_name, input_shape)]
qmod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
# count MACs
try:
    func = relay.testing.run_opt_pass(qmod["main"], relay.transform.InferType())
    compute_count = relay.analysis.get_total_mac_number(func)
    print('MACs = {:.2E} MACs'.format(compute_count))
except:
    print('MACs could not be calculated!')

with tvm.transform.PassContext(opt_level=3):
        #quantized_mod = relay.transform.ConvertLayout({'nn.conv2d': ['NHWC', 'OHWI']})(qmod)
        quantized_mod = relay.quantize.quantize(qmod, params,
                                            quantize_target=relay.transform.target.Float16QuantizeTarget())


# Quantization
# quantize model
#qmod = quantize_network(qmod, params)
#qmod =

# plot the weihgs after quantization from the relay IR moodel
# Create a single figure and axis for plotting
# fig, ax = plt.subplots(figsize=(10, 6))

# # Define a list of colors for the layers
# colors = ['skyblue', 'salmon', 'gold', 'lightgreen', 'violet']

# max_abs_weight = 0

# for layer_name, color in zip(conv_layers, colors):
#     layer = qmod.get_submodule(layer_name)
#     weights = layer.weight.data.cpu().numpy().flatten()  # Flatten to 1D for histogram
#     max_abs_weight = max(max_abs_weight, np.abs(weights).max())
#     ax.hist(weights, bins=50, density=True, alpha=0.7, color=color, label=f'{layer_name}')

# ax.set_xlim(left=-max_abs_weight, right=max_abs_weight)  # Center zero by setting symmetric xlim
# ax.set_title('Weight Distribution Across Layers')
# ax.set_xlabel('Weight Value')
# ax.set_ylabel('Density')
# ax.grid(axis='y', alpha=0.5)
# ax.legend()


# plt.savefig(f'weights_quant_tvm.png')

# Tracker RPC Configuration
# Step 1: Connect to the RPC Tracker
tracker_host = '10.100.100.131'  # The tracker's hostname or IP address
tracker_port = 9190  # The tracker's port
tracker_conn = rpc.connect_tracker(tracker_host, tracker_port)

# Step 2: Request a Remote Device Session
device_key = 'raspi5'  # The device key for the remote device
remote = tracker_conn.request(device_key)

# Tuning
target = tvm.target.Target("llvm -device=arm_cpu -model=bcm2712 -mtriple=aarch64-linux-gnu -mattr=+neon -mcpu=cortex-a76")
tasks = autotvm.task.extract_from_program(quantized_mod["main"], target=target, params=params)


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

for task in tasks:
    tuner = XGBTuner(task)
    tuner.tune(n_trial=tuning_option['n_trial'],
               early_stopping=tuning_option['early_stopping'],
               measure_option=tuning_option['measure_option'],
               callbacks=[autotvm.callback.log_to_file(tuning_option['log_filename']),
                          autotvm.callback.progress_bar(tuning_option['n_trial'])])

with autotvm.apply_history_best(tuning_option['log_filename']):
    with tvm.transform.PassContext(opt_level=3):
        quantized_mod = relay.transform.ConvertLayout({'nn.conv2d': ['NHWC', 'OHWI']})(qmod)
        quantized_mod = relay.quantize.quantize(quantized_mod, params,
                                            quantize_target=relay.transform.target.Float16QuantizeTarget())

        lib = relay.build(qmod, target=target, params=params)

# Save the library
tmp = utils.tempdir()
lib_fname = tmp.relpath("srcnn_quant.tar")
lib.export_library(lib_fname)


# Upload and load the library
remote.upload(lib_fname)
rlib = remote.load_module("srcnn_quant.tar")

# Create the remote runtime module
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))

module.set_input("input0", y)
plt.imshow(y.squeeze())
plt.savefig('input.png')

# Run the model
module.run()

# Get output
out = module.get_output(0)
print("Output shape:", out.shape)



# Save output image
preds = out.numpy().squeeze(0).squeeze(0) * 255.
output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
output = Image.fromarray(output)
output.save("output_quant.png")
print("Saved Inference Output")

#calculate psnr against bicubic and original image
# Calculate PSNR with scikit learn convert images to numpy before
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
import numpy as np
import os

psnr = peak_signal_noise_ratio(np.array(compare_image), np.array(output))
print(f'PSNR Original: {psnr:.2f} dB')
psnr_bicubic = peak_signal_noise_ratio(np.array(compare_image),image)
print(f'PSNR Bicubic: {psnr_bicubic:.2f} dB')




# Performance evaluation
ftimer = module.module.time_evaluator("run", dev, number=1, repeat=20)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))