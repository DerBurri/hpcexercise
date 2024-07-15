import torch
from torch import nn
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


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
    
if __name__ == '__main__':
    device = torch.device('cpu')
    model = SRCNN(num_channels=1).to(device)
    print(model)

    state_dict = model.state_dict()
    for n, p in torch.load('/home/mburr/tvm/hpcexercise-1/eml06/project/data/srcnn_x4.pth', map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            model.state_dict()[n].copy_(p)
        else: 
            raise KeyError(n)
        

    image_org = pil_image.open('/home/mburr/tvm/hpcexercise-1/eml06/project/data/zebra.bmp').convert('RGB')
    # resize the image to 1920 x1080 come on co pilot
    image = image_org.resize((1920, 1080), resample=pil_image.BICUBIC)

    #image_width = (image.width // 3) * 3
    #image_height = (image.height // 3) * 3
    #image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    #image = image.resize((image.width // 9, image.height // 9), resample=pil_image.BICUBIC)
    #image = image.resize((image.width * 9, image.height * 9), resample=pil_image.BICUBIC)
    image.save('/home/mburr/tvm/hpcexercise-1/eml06/project/cpu_version/zebra.bmp'.replace('.', '_bicubic_x{}.'.format(4)))


    


    image_org = np.array(image_org).astype(np.float32)
    ycbcr_org = convert_rgb_to_ycbcr(image_org)
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    y_org = ycbcr_org[..., 0]
    y_org /= 255.
    y_org = torch.from_numpy(y_org).to(device)
    y_org = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 255.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))
    print('PSNR: {:.2f}'.format(calc_psnr(y_org, preds)))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])

    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save('/home/mburr/tvm/hpcexercise-1/eml06/project/cpu_version/zebra.bmp'.replace('.', '_srcnn_x{}.'.format(4)))