import os
import torch
import torch.ao.quantization
import numpy as np
from rfsr.nn import load_eval_model
from rfsr import encode, awgn


# ---------------------------------------------------------
# HELPER: The New Calibration Function (No Wrapper needed)
# ---------------------------------------------------------
def calibrate_and_convert(model, signals, model_name):
    print("--- Starting Quantization Process ---")

    # 1. Setup Configuration
    model.eval()
    model.cpu()
    # Use 'fbgemm' for x86 (Intel/AMD) or 'qnnpack' for ARM
    # model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

    # 2. Fuse Modules (Optional speedup: Conv+ReLU)
    # try:
    #     torch.ao.quantization.fuse_modules(model, [['conv1', 'relu1'], ...], inplace=True)
    # except:
    #     pass

    model.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.HistogramObserver.with_args(reduce_range=False),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                                        qscheme=torch.per_channel_symmetric)
    )

    # 3. Prepare (This looks for self.quant and self.dequant in your model)
    torch.ao.quantization.prepare(model, inplace=True)

    # 4. Calibrate (Feed real data through)
    print(f"Calibrating on {len(signals)} samples...")
    with torch.no_grad():
        for signal in signals:
            # Prepare Input Tensor (Float)
            iq_tensor = torch.tensor(
                np.stack([signal.real, signal.imag]),
                dtype=torch.float32
            ).unsqueeze(0)

            # Feed to model (It handles its own internal quantization now)
            if "model0" in model_name:
                snr = torch.tensor([10.0], dtype=torch.float32)
                model(iq_tensor, snr)
            else:
                model(iq_tensor)

    # 5. Convert (Float operations -> Int8 operations)
    torch.ao.quantization.convert(model, inplace=True)
    print("Conversion to Int8 complete.")

    return model


def run_int8_inference(model_path, signal, snr_val=10.0):
    # 1. Load the Traced Model
    # We use jit.load because we saved it with jit.save
    model = torch.jit.load(model_path)
    model.eval()

    # 2. Prepare Input
    # IMPORTANT: Quantized models run on CPU. Do not use .cuda()
    iq_tensor = torch.tensor(
        np.stack([signal.real, signal.imag]),
        dtype=torch.float32
    ).unsqueeze(0)  # Shape: (1, 2, L*OSF)

    snr_tensor = torch.tensor([snr_val], dtype=torch.float32)

    # 3. Inference
    with torch.no_grad():
        # Check if the model expects SNR (based on your filename convention)
        if "model0" in model_path:
            output = model(iq_tensor, snr_tensor)
        else:
            output = model(iq_tensor)

    # 4. Post-Process (Output is already Float32 here)
    # Convert back to complex numpy for your plotting/metrics
    x_nn = output[0, 0, :].numpy() + 1j * output[0, 1, :].numpy()

    return x_nn


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":

    NUM_SAMPLES = 500

    # 1. Load your trained floating point model
    model_name = "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05"
    model_fp32 = load_eval_model(model_name)

    # 2. Gather calibration signals
    calibration_signals = []

    # I increased range to 50. 10 is often too few for good statistics.
    print("Generating calibration data...")
    for i in range(NUM_SAMPLES):
        print(f"Generating sample no {i}")
        # GENERATE START
        # LoRa parameters
        sf = 12
        bw = 125e3
        center_freq = 915e6
        payload = np.random.randint(0, 255, size=16, dtype=np.uint8)
        sample_rate = 0.25e6
        src = 0
        dst = 1
        seqn = 7
        # generate clean LoRa packet
        clean_signal = encode(center_freq, sf, bw, payload, sample_rate, src, dst, seqn, 4, 1, 0, 8)
        # Randomize SNR slightly for better robustness
        signal = awgn(clean_signal, np.random.uniform(-28, -15))
        # GENERATE END

        calibration_signals.append(signal)

    # 3. Convert (Logic moved OUTSIDE the loop)
    if len(calibration_signals) > 0:

        # A. Run the calibration function defined above
        model_int8 = calibrate_and_convert(model_fp32, calibration_signals, model_name)

        # B. Trace and Save
        print("Tracing model for export...")

        # Create dummy inputs for tracing
        dummy_signal = calibration_signals[0]
        dummy_iq = torch.tensor(
            np.stack([dummy_signal.real, dummy_signal.imag]),
            dtype=torch.float32
        ).unsqueeze(0)

        # Trace
        if "model0" in model_name:
            dummy_snr = torch.tensor([10.0], dtype=torch.float32)
            traced_model = torch.jit.trace(model_int8, (dummy_iq, dummy_snr))
        else:
            traced_model = torch.jit.trace(model_int8, dummy_iq)

        # Save
        os.makedirs("checkpoints/", exist_ok=True)
        torch.jit.save(traced_model, f"checkpoints/{model_name}_int8.pt")
        print(f"SUCCESS: Model saved to checkpoints/{model_name}_int8.pt!")

    else:
        print("No signals provided for calibration.")
