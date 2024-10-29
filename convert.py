from pydub import AudioSegment
import os

def flac_to_wav_and_save(source_folder):
    for dirpath, dirnames, filenames in os.walk(source_folder):
            for filename in filenames:
                if filename.endswith('.flac'):
                    flac_path = os.path.join(dirpath, filename)
                    print("FLAC Path:", flac_path)  # 打印查看FLAC文件路径
                    wav_path = os.path.join(dirpath, os.path.splitext(filename)[0] + '.wav')
                    audio = AudioSegment.from_file(flac_path, "flac")
                    # audio.export(wav_path, format="wav")
                    if os.path.exists(wav_path):
                            os.remove(flac_path)
                    print(f"remove:{flac_path}")

# 调用函数
source_folder = './data/LibriSpeech'
flac_to_wav_and_save(source_folder)