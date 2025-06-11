# models/vocoder/hifigan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                               dilation=dilation[0], padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[1], padding=self.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[2], padding=self.get_padding(kernel_size, dilation[2])))
        ])
        
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                               dilation=1, padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=self.get_padding(kernel_size, 1)))
        ])
        
    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size*dilation - dilation)/2)
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGAN(torch.nn.Module):
    def __init__(self, config):
        super(HiFiGAN, self).__init__()
        
        # 설정 파라미터
        self.upsample_rates = config.get('upsample_rates', [8, 8, 2, 2])
        self.upsample_kernel_sizes = config.get('upsample_kernel_sizes', [16, 16, 4, 4])
        self.upsample_initial_channel = config.get('upsample_initial_channel', 512)
        self.resblock_kernel_sizes = config.get('resblock_kernel_sizes', [3, 7, 11])
        self.resblock_dilation_sizes = config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        
        # 입력 투영
        self.conv_pre = weight_norm(nn.Conv1d(80, self.upsample_initial_channel, 7, 1, padding=3))
        
        # 업샘플링 레이어
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(self.upsample_rates, self.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(self.upsample_initial_channel//(2**i),
                                 self.upsample_initial_channel//(2**(i+1)),
                                 k, u, padding=(k-u)//2)))
        
        # ResBlock
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # 출력 레이어
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(3):  # 3개의 다른 커널 사이즈
                if xs is None:
                    xs = self.resblocks[i*3+j](x)
                else:
                    xs += self.resblocks[i*3+j](x)
            x = xs / 3
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        """사전 훈련된 모델 로드"""
        if config is None:
            config = {
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4],
                'upsample_initial_channel': 512,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            }
        
        model = cls(config)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['generator'])
        return model