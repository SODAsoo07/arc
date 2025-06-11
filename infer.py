import os
import json
import argparse
import torch
import torchaudio
from transformers import HubertModel

from models.evc.durflex import DurFlexEVC
from tasks.evc.evc_utils import VocoderInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams
from utils.audio.io import save_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config", type=str, default="./configs/exp/durflex_evc_esd.yaml"
    )
    parser.add_argument("--src_wav", type=str, default="./sample/0011_000021.wav")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    hparams = set_hparams(args.config)
    
    # ===== work_dir 문제 해결을 위한 코드 추가 시작 =====
    print(f"기존 work_dir 값: '{hparams.work_dir}'")
    print(f"hparams 타입: {type(hparams)}")
    print(f"work_dir 길이: {len(hparams.work_dir)}")
    
    # work_dir이 빈 문자열이거나 None인 경우 올바른 경로로 설정
    if not hparams.work_dir or hparams.work_dir.strip() == "":
        # setattr을 사용해서 속성 값 변경
        setattr(hparams, 'work_dir', "/content/drive/MyDrive/EVC/Save_ckpt/DurFlex")
        print(f"work_dir이 비어있어서 새로 설정됨: {hparams.work_dir}")
    
    # 실제 사용할 work_dir 경로 설정
    actual_work_dir = hparams.work_dir if hparams.work_dir else "/content/drive/MyDrive/EVC/Save_ckpt/DurFlex"
    print(f"실제 사용할 work_dir: {actual_work_dir}")
    
    # 체크포인트 디렉토리 존재 확인
    if os.path.exists(actual_work_dir):
        print(f"체크포인트 디렉토리 존재 확인: {actual_work_dir}")
        files = os.listdir(actual_work_dir)
        print(f"디렉토리 내 파일들: {files}")
        
        # .ckpt 파일들만 필터링해서 표시
        ckpt_files = [f for f in files if f.endswith('.ckpt')]
        if ckpt_files:
            print(f"체크포인트 파일들: {ckpt_files}")
        else:
            print("체크포인트 파일(.ckpt)을 찾을 수 없습니다!")
    else:
        print(f"경고: 체크포인트 디렉토리가 존재하지 않습니다: {actual_work_dir}")
        print("경로를 다시 확인해주세요.")
    # ===== work_dir 문제 해결을 위한 코드 추가 끝 =====
    
    os.makedirs(args.save_dir, exist_ok=True)
    sample_rate = hparams["audio_sample_rate"]
    spk_dict = json.load(
        open(os.path.join(hparams["processed_data_dir"], "spk_map.json"))
    )
    emo_dict = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}

    hubert = HubertModel.from_pretrained(
        hparams["hubert_model"], output_hidden_states=True
    ).cuda()
    model = DurFlexEVC(hparams["n_units"], hparams).cuda()
    
    # 체크포인트 로딩 시도 (actual_work_dir 사용)
    try:
        load_ckpt(model, actual_work_dir, "model")
        print("체크포인트 로딩 성공!")
    except Exception as e:
        print(f"체크포인트 로딩 실패: {e}")
        print("다음을 확인해주세요:")
        print("1. 체크포인트 파일이 올바른 경로에 있는지")
        print("2. 파일명이 올바른지 (보통 model_ckpt_steps_*.ckpt 형식)")
        print("3. load_ckpt 함수가 올바른 파일명 패턴을 찾고 있는지")
        
        # 추가 디버깅: load_ckpt 함수가 찾는 파일명 패턴 확인
        print(f"\n디버깅 정보:")
        print(f"전달된 work_dir: {actual_work_dir}")
        print(f"전달된 prefix: 'model'")
        raise e
    
    vocoder = VocoderInfer(hparams)

    basename = os.path.basename(args.src_wav).replace(".wav", "")
    print(f"처리할 파일명: {basename}")
    
    # 파일명 형식 검사 및 처리
    if "_" in basename and len(basename.split("_")) >= 2:
        # 기존 형식: 0011_000021.wav
        parts = basename.split("_")
        spk = parts[0]
        try:
            item_id = int(parts[1])
            if item_id < 351:
                label = "Neutral"
            elif item_id < 701:
                label = "Angry"
            elif item_id < 1051:
                label = "Happy"
            elif item_id < 1401:
                label = "Sad"
            else:
                label = "Surprise"
        except ValueError:
            print(f"경고: item_id를 숫자로 변환할 수 없습니다: {parts[1]}")
            label = "Neutral"  # 기본값
    else:
        # 새로운 형식: JpsampleNa.wav 등
        print(f"파일명이 예상 형식(spk_itemid)과 다릅니다: {basename}")
        print("기본값을 사용합니다.")
        spk = "0001"  # 기본 화자 ID
        label = "Neutral"  # 기본 감정
    
    print(f"화자: {spk}, 감정: {label}")

    y, sr = torchaudio.load(args.src_wav)
    x = hubert(y.cuda()).hidden_states[-1]
    
    # spk_dict에서 화자 ID 확인 및 안전 처리
    print(f"추출된 화자 ID: '{spk}'")
    
    if spk in spk_dict:
        spk_id = spk_dict[spk]
        print(f"화자 ID 매핑: {spk} -> {spk_id}")
    else:
        print(f"경고: 화자 '{spk}'가 spk_dict에 없습니다.")
        print(f"사용 가능한 화자들: {list(spk_dict.keys())}")
        
        # 숫자 형태의 화자 ID 변환 시도
        available_speakers = list(spk_dict.keys())
        
        # 1. 정확한 매칭 시도 (문자열을 4자리로 패딩)
        padded_spk = spk.zfill(4)  # "0" -> "0000"
        if padded_spk in spk_dict:
            spk = padded_spk
            spk_id = spk_dict[spk]
            print(f"패딩된 화자 ID 사용: {spk} -> {spk_id}")
        
        # 2. 숫자 기반 매핑 시도
        elif any(speaker.isdigit() for speaker in available_speakers):
            # 숫자로만 구성된 화자 ID들 찾기
            numeric_speakers = [s for s in available_speakers if s.isdigit()]
            if numeric_speakers:
                # 가장 가까운 숫자 찾기 또는 첫 번째 숫자 화자 사용
                try:
                    target_num = int(spk)
                    # 정확한 숫자 매칭
                    if str(target_num) in numeric_speakers:
                        spk = str(target_num)
                    else:
                        # 첫 번째 숫자 화자 사용
                        spk = numeric_speakers[0]
                except ValueError:
                    spk = numeric_speakers[0]
                
                spk_id = spk_dict[spk]
                print(f"숫자 기반 화자 ID 사용: {spk} -> {spk_id}")
            else:
                # 3. 첫 번째 화자를 기본값으로 사용
                spk = available_speakers[0]
                spk_id = spk_dict[spk]
                print(f"기본 화자 사용: {spk} -> {spk_id}")
        else:
            # 3. 첫 번째 화자를 기본값으로 사용
            spk = available_speakers[0]
            spk_id = spk_dict[spk]
            print(f"기본 화자 사용: {spk} -> {spk_id}")
    
    src_emo = emo_dict[label]
    spk_id = torch.LongTensor([spk_id]).cuda()
    src_emo = torch.LongTensor([src_emo]).cuda()
    for k, v in emo_dict.items():

        tgt_emo = torch.LongTensor([v]).cuda()
        output = model(
            x, spk_id=spk_id, emotion_id=src_emo, tgt_emotion_id=tgt_emo, infer=True
        )
        mel_out = output["mel_out"]
        wav_out = vocoder.spec2wav(mel_out[0].cpu())
        save_wav(wav_out, f"{args.save_dir}/{basename}_{k}.wav", sample_rate)
        print(f"{basename} converted to {k}")