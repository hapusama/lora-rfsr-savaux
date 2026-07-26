"""公开 RF-SR 权重的最小推理示例。

本脚本不训练模型，也不读取 OTA 数据集。它生成一个 2 MSPS 的干净 LoRa
packet、添加 AWGN、降采样到 250 kSPS，再调用公开四层 CNN 输出 1 MSPS
复数 IQ，最后交给普通 LoRa 解码器。

接入 USRP 文件时，应将 ``signal_ds`` 替换为已经同步/切片后的 250 kSPS
complex IQ；网络输入仍需转换为 ``[B, 2, L]`` 的 float32 张量。
"""

import torch
import numpy as np

from rfsr import decode, encode, awgn
from rfsr.nn import load_eval_model


def main():
    # 加载仓库自带的浮点权重；load_eval_model 会解析名字中的模型和 OSF。
    model_name = "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05"
    model = load_eval_model(f"{model_name}")

    # 生成一个已知 payload 的 2 MSPS 干净 LoRa packet。
    sf, bw, sample_rate = 12, 125e3, 2e6
    center_freq, src, dst, seqn, payload = 915e6, 0, 1, 7, np.random.randint(0, 255, size=16, dtype=np.uint8)
    signal = encode(center_freq, sf, bw, payload, sample_rate, src, dst, seqn, 4, 1, 0, 8)

    # 对整段 IQ（包括 packet 前的静默段）加入合成 AWGN。
    snr = -22 # np.random.uniform(-35, 10)
    signal = awgn(signal, snr)

    # 模拟低速 ADC：2 MSPS 每 8 点抽 1 点，得到 250 kSPS 输入。
    DS = 8  # DS to 0.25e6
    signal_ds = signal[::DS]

    # RF-SR：polyphase 插值 4 倍，再由 CNN 预测残差，输出 1 MSPS。
    UPS = 4 # upsampling factor
    with torch.no_grad():
        iq_tensor = torch.tensor(
            np.stack([signal_ds.real, signal_ds.imag]),
            dtype=torch.float32
        ).unsqueeze(0)  # adds batch dim
        output = model(iq_tensor, snr)  # shape (1, 2, L*OSF)
    signal_nn = output[0, 0, :].cpu().numpy() + 1j * output[0, 1, :].cpu().numpy()

    # 使用普通 LoRa 解码器检查 RF-SR 输出能否还原出一个 packet。
    try:
        x = decode(signal_nn, sf, bw, (UPS * sample_rate) / DS)
    except:
        x = []
    if len(x) != 1:
        print(f"Number of decoded packets: {len(x)} != 1 (actual:{len(x)})-> Fail")
    else:
        print("Success")


if __name__ == "__main__":
    main()
