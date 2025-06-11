import librosa
import soundfile as sf
import os
from pathlib import Path

def resample_wav(input_file, output_file, target_sr=16000):
    """
    Resamples a WAV file to a specified sample rate.

    Args:
        input_file (str): Path to the input WAV file.
        output_file (str): Path to save the resampled WAV file.
        target_sr (int): Desired sample rate (default 16000 Hz).
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_file, sr=None)

        # Resample if the original sample rate is not the target
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        else:
            y_resampled = y

        # Save the resampled audio
        sf.write(output_file, y_resampled, target_sr)

        print(f"✓ Resampled: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(input_file)}: {e}")
        return False

def batch_resample_folder(input_folder, output_folder=None, target_sr=16000):
    """
    Batch resamples all WAV files in a folder.

    Args:
        input_folder (str): Path to the folder containing input WAV files.
        output_folder (str): Path to save resampled files. If None, creates 'resampled' subfolder.
        target_sr (int): Desired sample rate (default 16000 Hz).
    """
    input_path = Path(input_folder)
    
    # Check if input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Set output folder
    if output_folder is None:
        output_path = input_path / "resampled"
    else:
        output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
    
    if not wav_files:
        print(f"No WAV files found in '{input_folder}'")
        return
    
    print(f"Found {len(wav_files)} WAV files to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_path}")
    print(f"Target sample rate: {target_sr} Hz")
    print("-" * 50)
    
    # Process each file
    success_count = 0
    for wav_file in wav_files:
        output_file = output_path / wav_file.name
        if resample_wav(str(wav_file), str(output_file), target_sr):
            success_count += 1
    
    print("-" * 50)
    print(f"Processing complete: {success_count}/{len(wav_files)} files processed successfully")

if __name__ == '__main__':
    # 사용 예시
    input_folder = './train_Data/0011/Sad' # 입력 폴더 경로
    output_folder = './train_Data/1611/Sad'# 출력 폴더 경로 (선택사항)
    target_sample_rate = 16000  # 목표 샘플레이트
    
    # 출력 폴더를 지정하지 않으면 입력 폴더 안에 'resampled' 폴더가 생성됩니다
    batch_resample_folder(input_folder, output_folder, target_sample_rate)
    
    # 또는 출력 폴더 없이 실행 (입력 폴더 안에 'resampled' 폴더 생성)
    # batch_resample_folder(input_folder, target_sr=target_sample_rate)