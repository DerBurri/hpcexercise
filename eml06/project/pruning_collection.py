def prune_weights_by_threshold(model, threshold):

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name != 'conv1':  # Skip the first layer
                weight = module.weight.data.clone()  # Create a copy to avoid modifying original weights
                mask = torch.abs(weight) >= threshold  # Mask of weights above the threshold
                module.weight.data *= mask  # Apply the mask to zero out weights below the threshold
                if False:  # Set to True if you want permanent removal
                    module.weight = nn.Parameter(module.weight.data.masked_select(mask))


# Example usage:
threshold = 0.001  # Prune by Threshhold Value
prune_weights_by_threshold(model, threshold)


#--------------------------------------------------------------------------------------------------------------------------------------------

def prune_channels_by_fraction(model, fraction):

    with torch.no_grad():
        prunable_layers = [model.conv2, model.conv3] # Skip the first layer

        # Gather channel importance scores (L1 norm of weights) for prunable layers
        all_channel_norms = torch.cat([torch.sum(torch.abs(layer.weight.data), dim=(0, 2, 3)) for layer in prunable_layers])
        num_channels_to_prune = int(fraction * len(all_channel_norms))
        threshold, _ = torch.kthvalue(all_channel_norms, num_channels_to_prune)  # Find the threshold

        # Prune channels in each prunable layer
        for layer in prunable_layers:
            channel_l1_norms = torch.sum(torch.abs(layer.weight.data), dim=(0, 2, 3))
            mask = channel_l1_norms >= threshold

            # Apply the mask to zero out channels below the threshold
            layer.weight.data *= mask.view(1, -1, 1, 1)
            if layer.bias is not None:  # Some layers might not have bias
                layer.bias.data *= mask

            # (Optional) Remove pruned channels permanently and adjust following layers 
            if False:  # Set to True if you want permanent removal
                # Remove pruned channels from current layer
                pruned_weight = layer.weight.data.masked_select(mask.view(1, -1, 1, 1))
                pruned_weight = pruned_weight.view(-1, mask.sum(), layer.kernel_size[0], layer.kernel_size[1])
                layer.weight = nn.Parameter(pruned_weight)

                # Adjust the next layer's input channels (if there is a next layer)
                next_layer = prunable_layers[prunable_layers.index(layer) + 1] if prunable_layers.index(layer) < len(prunable_layers) - 1 else None
                if next_layer is not None:
                    pruned_next_layer_weight = next_layer.weight.data[:, mask, :, :]
                    next_layer.weight = nn.Parameter(pruned_next_layer_weight)

fraction = 0.2  # Prune by % of channels
prune_channels_by_fraction(model, fraction)

#--------------------------------------------------------------------------------------------------------------------------------------------

def prune_layers_by_name(model, layer_names):

    for name in layer_names:
        if hasattr(model, name):  # Check if the layer exists in the model
            module = getattr(model, name)
            if isinstance(module, torch.nn.Conv2d):
                if False: # Set to True if you want permanent removal
                    setattr(model, name, torch.nn.Identity())  # Replace with Identity layer
                    print(f"Layer '{name}' permanently removed.")
                else:
                    module.weight.data.zero_()  # Zero out the weights
                    if module.bias is not None:
                        module.bias.data.zero_()
                    print(f"Layer '{name}' pruned (weights zeroed).")
            else:
                raise ValueError(f"Layer '{name}' is not a convolutional layer and cannot be pruned.")
        else:
            raise ValueError(f"Layer '{name}' not found in the model.")


# Example usage:
layers_to_prune = ['conv2'] # Prune Layer by Name
prune_layers_by_name(model, layers_to_prune)