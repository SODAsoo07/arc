import importlib
import torch
from utils.commons.hparams import hparams, set_hparams
import numpy as np


# tasks/evc/evc_utils.py 수정 부분
class VocoderInfer:
    def __init__(self, hparams):
        self.hparams = hparams
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 보코더 설정
        vocoder_ckpt = hparams['vocoder_ckpt']
        vocoder_cls = hparams.vocoder_cls if hasattr(hparams, 'vocoder_cls') else 'models.vocoder.hifigan.HiFiGAN'
        
        # 모델 클래스 동적 로드
        if 'hifigan' in vocoder_cls.lower():
            from models.vocoder.hifigan import HiFiGAN
            config = {
                'upsample_rates': [8, 8, 2, 2],
                'upsample_kernel_sizes': [16, 16, 4, 4], 
                'upsample_initial_channel': 512,
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
            }
            self.model = HiFiGAN(config)
        else:
            # 기존 BigVGAN 등 다른 보코더
            vocoder_cls = locate(vocoder_cls)
            self.model = vocoder_cls(hparams)
        
        # 체크포인트 로드 (안전한 방식)
        checkpoint_dict = torch.load(vocoder_ckpt, map_location='cpu')
        
        # state_dict 키 확인 및 로드
        if 'generator' in checkpoint_dict:
            state_dict = checkpoint_dict['generator']
        elif 'model' in checkpoint_dict:
            state_dict = checkpoint_dict['model']
        else:
            state_dict = checkpoint_dict
            
        # 호환되지 않는 키 필터링
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Shape mismatch for {key}: {model_state_dict[key].shape} vs {value.shape}")
            else:
                print(f"Key not found in model: {key}")
        
        # 필터링된 state_dict 로드
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        
    def spec2wav(self, mel):
        """멜 스펙트로그램을 오디오로 변환"""
        device = next(self.model.parameters()).device
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        mel = mel.to(device)
        
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)  # 배치 차원 추가
            
        with torch.no_grad():
            wav = self.model(mel)
            if wav.dim() == 3:
                wav = wav.squeeze(1)  # 채널 차원 제거
        return wav.cpu().numpy()


def parse_dataset_configs():
    max_tokens = hparams["max_tokens"]
    max_sentences = hparams["max_sentences"]
    max_valid_tokens = hparams["max_valid_tokens"]
    if max_valid_tokens == -1:
        hparams["max_valid_tokens"] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams["max_valid_sentences"]
    if max_valid_sentences == -1:
        hparams["max_valid_sentences"] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences
