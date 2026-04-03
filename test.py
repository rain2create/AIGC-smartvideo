# test_paddle.py
import sys
sys.path.insert(0, '/mnt_liuhuan/search_public/xby/smart_video')

from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

# 测试本地模型
am_ckpt = "/root/.paddlespeech/models/fastspeech2_male_zh_ckpt_1.4.0/snapshot_iter_76000.pdz"
print(f"检查模型文件: {Path(am_ckpt).exists()}")

try:
    from paddlespeech.cli.tts.infer import TTSExecutor
    tts = TTSExecutor()
    
    # 使用本地模型参数（通过初始化参数而非命令行）
    tts(
        text="你好，测试本地模型",
        output="/tmp/test_local.wav",
        am="fastspeech2_male",
        am_config="/root/.paddlespeech/models/fastspeech2_male_zh_ckpt_1.4.0/default.yaml",
        am_ckpt=am_ckpt,
        phones_dict="/root/.paddlespeech/models/fastspeech2_male_zh_ckpt_1.4.0/phone_id_map.txt",
        voc="pwgan_aishell3",
        voc_config="/root/.paddlespeech/models/pwgan_aishell3_ckpt_0.5/default.yaml",
        voc_ckpt="/root/.paddlespeech/models/pwgan_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz"
    )
    print("✓ 测试成功！可以修改主脚本了")
except Exception as e:
    print(f"✗ 失败: {e}")
    print("需要改用 edge-tts 方案")