import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse


class Encoder1(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.dwt1 = DWTForward(J=2, mode='zero', wave=wavelet)
        self.dwt2 = DWTForward(J=1, mode='zero', wave=wavelet)

    def forward(self, x):
        assert x.shape[1] == 3, x.shape
        x_1 = x[:, :1, ...]
        x_2 = x[:, 1:2, ...]
        x_3 = x[:, 2:, ...]
        x_2 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x_3 = F.interpolate(x_3, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        y0, (yh0, yh1) = self.dwt1(x_1)
        y1, (yh2,) = self.dwt2(x_2)
        y2, (yh3,)  = self.dwt2(x_3)
        yh0 = rearrange(yh0, 'b n c (d h) (e w) -> b (n c d e) h w', d=2, e=2)
        yh1 = rearrange(yh1, 'b n c h w -> b (n c) h w')
        yh2 = rearrange(yh2, 'b n c h w -> b (n c) h w')
        yh3 = rearrange(yh3, 'b n c h w -> b (n c) h w')
        print(y0.shape,y1.shape,y2.shape)
        print(yh0.shape, yh1.shape, yh2.shape, yh3.shape)
        y = torch.cat([y0, yh0, yh1, y1, yh2, y2, yh3], dim=1)
        # 24 = 1 + 12 + 3 + 1 + 3 + 1 + 3
        return y


class Decoder1(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.idwt = DWTInverse(mode='zero', wave=wavelet)

    def forward(self, h):
        assert h.shape[1] == 24
        y0 = h[:, :1, ...] # 1
        yh0 = h[:, 1:13, ...]  # 12
        yh1 = h[:, 13:16, ...] # 3
        y1 = h[:, 16:17, ...] # 1
        yh2 = h[:, 17:20, ...] # 3
        y2 = h[:, 20:21, ...] # 1
        yh3 = h[:, 21:, ...] # 3
        yh0 = rearrange(yh0, 'b (n c d e) h w -> b n c (d h) (e w)', n=1, c=3, d=2, e=2)
        yh1 = rearrange(yh1, 'b (n c) h w -> b n c h w', n=1, c=3)
        yh2 = rearrange(yh2, 'b (n c) h w -> b n c h w', n=1, c=3)
        yh3 = rearrange(yh3, 'b (n c) h w -> b n c h w', n=1, c=3)
        print(y0.shape,y1.shape,y2.shape)
        print(yh0.shape, yh1.shape, yh2.shape, yh3.shape)

        x_1, x_2, x_3 = self.idwt((y0, (yh0, yh1))), self.idwt((y1, (yh2,))), self.idwt((y2, (yh3,)))
        x_2 = F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        return x


class WaveEncoder(nn.Module):
    def __init__(self, level=3, wavelet='haar'):
        super().__init__()
        self.level = level
        self.dwt = DWTForward(J=level, mode='zero', wave=wavelet)

    def forward(self, x):
        assert x.shape[1] == 3, x.shape
        ## x shape: NCHW
        ll, hh = self.dwt(x)
        # ll shape: NChw, h = H/2**level, w = W/2**level
        # hh shape: NC3hw, 3： (H V D)
        assert len(hh) == self.level
        y = ll
        for l in range(self.level):
            h = hh[l]
            h = rearrange(
                h, 
                'n c d (p h) (q w) -> n (c d p q) h w', 
                p=2**(self.level - l - 1), 
                q=2**(self.level - l - 1)
            )
            y = torch.cat([y, h], dim=1)
        return y

def calculate_channel(level, channel=3):
    list_channel = [channel]
    cc = channel
    for l in range(level):
        cc += (channel * 3 * 4 ** (level - l  -1))
        list_channel.append(cc)
    return list_channel


class WaveDecoder(nn.Module):
    def __init__(self, level = 3, wavelet='haar', channel=3):
        super().__init__()
        self.level = level
        self.channel = channel
        # calculate_channel according level
        self.channels = calculate_channel(level, channel=channel)
        self.idwt = DWTInverse(mode='zero', wave=wavelet)

    def forward(self, h):
        assert h.shape[1] == 3 * 4 ** self.level, h.shape
        list_subbands = []
        ll = h[:, :self.channels[0], ...]
        list_subbands.append(ll)
        list_subhigh = []
        for l in range(self.level):
            h_ = h[:, self.channels[l]:self.channels[l+1], ...]
            h_ = rearrange(
                h_,
                'n (c d p q) h w -> n c d (p h) (q w)',
                c = self.channel, d=3, 
                p = 2 ** (self.level - l - 1), 
                q = 2 ** (self.level - l - 1), 
            )
            list_subhigh.append(h_)
        list_subbands.append(list_subhigh)
        x = self.idwt(list_subbands)
        return x
        

if __name__ == '__main__':
    # test
    x = torch.randn(1, 3, 256, 256)
    encoder = WaveEncoder(level=2)
    y = encoder(x)
    print(y.shape)


    # test decoder
    print("decoder:.....................")
    decoder = WaveDecoder(level=2, channel=3)
    xo = decoder(y)
    print(xo.shape)
    # difference
    print(torch.abs((xo-x)).mean())

    # test image 
    # import cv2 
    # import numpy as np
    # path = 'data/inpainting_examples/6458524847_2f4c361183_k.png'
    # import os 
    # assert os.path.exists(path)
    # img_rgb = cv2.imread(path)
    # img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb)
    # img_t = torch.from_numpy(img_ycbcr).permute(2, 0, 1).unsqueeze(0).float()/255.0
    # print(img_t.shape)
    # y1 = encoder(img_t)
    # print(y1.shape)
    # xo1 = decoder(y1)
    # print(xo1.shape)
    # img_o = xo1.permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()
    # img_o = (img_o*255.0).astype(np.uint8)
    # img_o = cv2.cvtColor(img_o, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite('test2.png', img_o)
