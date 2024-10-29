from pydub import AudioSegment
import os

def flac_to_wav_and_save(source_folder):
    for dirpath, dirnames, filenames in os.walk(source_folder):
            for filename in filenames:
                if filename.endswith('.flac'):
                    try:
                        # 构建完整的 FLAC 文件路径
                        flac_path = os.path.join(dirpath, filename)
                        print("FLAC Path:", flac_path)  # 打印查看FLAC文件路径

                        # 构建 WAV 文件路径
                        wav_path = os.path.join(dirpath, os.path.splitext(filename)[0] + '.wav')

                        # 读取并转换音频文件
                        audio = AudioSegment.from_file(flac_path, "flac")
                        audio.export(wav_path, format="wav")

                        print(f"Converted {filename} to WAV format.")
                    except Exception as e:
                        print(f"Failed to convert {filename}: {str(e)}")

# 调用函数
source_folder = './data/LibriSpeech'
flac_to_wav_and_save(source_folder)