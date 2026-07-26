import os
import json5
import torch
import argparse
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from rfsr import decode, encode, awgn
from rfsr.nn import load_eval_model, run_int8_inference


def plot_per_vs_snr(set="synth_nn", SNR=(-27, -10)):
    # set style
    plt.style.use(['science', 'ieee'])  # or 'nature' if you prefer
    mpl.rcParams["text.usetex"] = False

    # Load results
    with open("results/per_vs_fs.json", "r") as f:
        data = json5.load(f)

    # Prepare plot
    plt.figure(figsize=(8, 4))

    # Define sampling rate labels (and convert keys like "2e6" to readable form)
    if set == "synth_nn":
        fs_labels = {
            "stdlora_2e6": "2 MSPS",  # "Synth 2 MSPS",
            "stdlora_1e6": "1 MSPS",  # "Synth 1 MSPS",
            "stdlora_0.5e6": "0.5 MSPS",  # "Synth 0.5 MSPS",
            "stdlora_0.25e6": "0.25 MSPS",  # "Synth 0.25 MSPS",
            "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05_0.25e6": "RF-SR 0.25 MSPS",  # "model0v0",
            "rfsrint8quant_0.25e6": "RF-SR INT8 0.25 MSPS",  # "model0v0",
            # "synth_loratrimmer1e6": "LoRaTrimmer 1e6",
            # "synth_loratrimmer0.25e6": "LoRaTrimmer 0.25e6",
            # "synth_nelora1e6": "NeLoRa 1e6",
            # "synth_gloriphy1e6": "GLoRiPHY 1e6",
        }
    else:
        raise RuntimeError(f"No valid set selected: {set}")

    # Define more "scientific" colors + different line styles for BW-friendly output
    styles = [
        ("#0072B2", "-", None),  # 1. Dark Blue, solid
        ("#D55E00", "--", None),  # 2. Vermillion, dashed
        ("#009E73", "-.", None),  # 3. Green, dash-dot
        ("#CC79A7", ":", None),  # 4. Bluish Pink, dotted
        ("#E69F00", ":", 'x'),  # 5. Yellow ,dotted
        ("#56B4E9", ':', 'o'),  # 6. Sky Blue, solid
        ("#999999", "-.", None),  # 7. Gray, dash-dot
        ("#263A50", ":", None),  # 8. Dark Slate, dotted
        ("#F0E442", "-", None),  # 9. Yellow, solid
        ("#B8860B", "--", None),  # 10. Dark Gold, dashed
    ]

    FONTSIZE = 17
    # Set all font sizes globally
    plt.rcParams.update({
        'font.size': FONTSIZE,  # Base font size
        'axes.titlesize': FONTSIZE,  # Title font size
        'axes.labelsize': FONTSIZE,  # X and Y label font size
        'xtick.labelsize': FONTSIZE,  # X-axis tick label size
        'ytick.labelsize': FONTSIZE,  # Y-axis tick label size
        'legend.fontsize': FONTSIZE * 0.8,  # Legend font size
        'figure.titlesize': FONTSIZE  # Main figure title
    })

    for (fs_key, (color, linestyle, marker)) in zip(fs_labels.keys(), styles):
        entries = data.get(fs_key, [])
        if not entries:
            continue

        # Split into SNR and success
        snrs, successes = zip(*entries)
        snrs = np.array(snrs)
        successes = np.array(successes)

        # Define bins for SNR averaging
        bins = np.linspace(np.min(snrs), np.max(snrs), 50)
        per = []
        snr_centers = []

        for i in range(len(bins) - 1):
            mask = (snrs >= bins[i]) & (snrs < bins[i + 1])
            if np.sum(mask) == 0:
                continue
            snr_centers.append((bins[i] + bins[i + 1]) / 2)
            per.append(1 - np.mean(successes[mask]))  # PER = 1 - success rate

        # Plot PER vs SNR
        plt.plot(snr_centers, per, label=fs_labels[fs_key],
                 color=color, linestyle=linestyle, marker=marker, linewidth=1.6)

    plt.xlabel("SNR [dB]")
    plt.ylabel("Packet Error Rate")
    plt.title("LoRa PER vs. SNR ")
    plt.legend(frameon=True, facecolor='white', edgecolor='black', title="")

    plt.tight_layout()
    plt.xlim(SNR)
    os.makedirs("results/", exist_ok=True)
    plt.savefig(f"results/per_vs_snr_{set}.png")
    plt.show()


if __name__ == "__main__":

    # Evaluate the packet error rate
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and display the plot"
    )

    parser.add_argument(
        "--synth_nn",
        action="store_true",
        help=""
    )

    # Add the specific argument
    parser.add_argument(
        '--set',
        type=str,
        choices=[
            'quant1e6',
            'quant2e6',
            'quant0.5e6',
            'quant0.25e6',
            'synth_nn',
            'ota_nn',
        ],
        default="synth_nn",
        help="Specify the dataset or configuration to use."
    )
    args = parser.parse_args()

    # switch by user selection
    if args.plot:
        plot_per_vs_snr(args.set)
    elif args.synth_nn:


        NUM_SAMPLES = 1000
        SNR = (-35, 0)

        sf = 12
        bw = 125e3
        sample_rate = 2e6

        # Path to the file,         # Keys to ensure
        path = "results/per_vs_fs.json"
        os.makedirs("results/", exist_ok=True)
        if os.path.exists(path):  # Load if exists
            with open(path, "r") as f:
                result = json5.load(f)
        else:
            result = {}


        #############
        # Evalutation Loop
        np.random.seed(42)  # new seed !=42 -> new synthetic dataset!
        for i in range(NUM_SAMPLES):  # we generate fresh data (new seed!) -> no problem with training / test dataset

            ######
            # Prepare LoRa Sample w/ Noise

            # LoRa parameters
            center_freq = 915e6
            payload = np.random.randint(0, 255, size=16, dtype=np.uint8)
            src = 0
            dst = 1
            seqn = 7
            # generate clean LoRa packet
            signal = encode(center_freq, sf, bw, payload, sample_rate, src, dst, seqn, 4, 1, 0, 8)

            # add noise
            snr = np.random.uniform(*SNR)
            print(f"snr={snr}")
            signal = awgn(signal, snr)


            ##########
            # Std LoRa + Downsampling
            for N in [1, 2, 4, 8]:  # 1x->2MSPS, 2x->1MSPS, 4x->500KSPS, 8x->250KSPS

                pre = "stdlora_"

                print(f"Using Std LoRa DS={N}, SNR={snr}, round={i}")

                # Downsampled signal
                signal_ds = signal[::N]

                successful = False
                try:
                    x = decode(signal_ds, sf, bw, sample_rate / N)
                except:
                    x = []
                if len(x) != 1:
                    print(f"Number of decoded packets: {len(x)} != 1 (actual:{len(x)})-> skip")
                else:
                    print("Success")
                    successful = True

                # store result
                if N == 1:
                    key = f"{pre}2e6"
                elif N == 2:
                    key = f"{pre}1e6"
                elif N == 4:
                    key = f"{pre}0.5e6"
                elif N == 6:
                    key = f"{pre}0.33e6"
                elif N == 8:
                    key = f"{pre}0.25e6"
                if key not in result:
                    result[key] = []
                result[key].append([snr, successful])

            #############
            # RFSR INT8 Quant
            quantmodel_name = "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05_int8"
            pre = "rfsrint8quant_"
            for DS in [8]:

                print(f"Using RF-SR Quant with DS={DS}, SNR={snr}, round={i}")

                # Downsampled signal
                signal_ds = signal[::DS]

                UPS = 4
                signal_nn = run_int8_inference(f"checkpoints/{quantmodel_name}.pt", signal_ds, snr_val=snr)

                # compute packet error rate
                successful = False
                # try:
                x = decode(signal_nn, sf, bw, (UPS * sample_rate) / DS)
                # except Exception as e:
                #     print(f"An unexpected error occurred: {e}")
                # x = []
                if len(x) != 1:
                    print(f"Number of decoded packets: {len(x)} != 1 (actual:{len(x)})-> skip")
                else:
                    print("Success")
                    successful = True

                # store result
                if DS == 1:
                    key = f"{pre}2e6"
                elif DS == 2:
                    key = f"{pre}1e6"
                elif DS == 4:
                    key = f"{pre}0.5e6"
                elif DS == 6:
                    key = f"{pre}0.33e6"
                elif DS == 8:
                    key = f"{pre}0.25e6"
                if key not in result:
                    result[key] = []
                result[key].append([snr, successful])

            #############
            # RFSR
            model_name = "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05"
            pre = f"{model_name}_"
            for DS in [8]: # DS to 0.25e6
                # load model
                model = load_eval_model(f"{model_name}")

                # Downsampled signal
                signal_ds = signal[::DS]

                # upsample using the neural network model
                UPS = 4
                iq_tensor = torch.tensor(
                    np.stack([signal_ds.real, signal_ds.imag]),
                    dtype=torch.float32
                ).unsqueeze(0)  # adds batch dim
                with torch.no_grad():
                    output = model(iq_tensor, snr)  # shape (1, 2, L*OSF)
                signal_nn = output[0, 0, :].cpu().numpy() + 1j * output[0, 1, :].cpu().numpy()

                # compute packet error rate
                successful = False
                try:
                    x = decode(signal_nn, sf, bw, (UPS * sample_rate) / DS)
                except:
                    x = []
                if len(x) != 1:
                    print(f"Number of decoded packets: {len(x)} != 1 (actual:{len(x)})-> skip")
                else:
                    print("Success")
                    successful = True

                # store result
                if DS == 1:
                    key = f"{pre}2e6"
                elif DS == 2:
                    key = f"{pre}1e6"
                elif DS == 4:
                    key = f"{pre}0.5e6"
                elif DS == 6:
                    key = f"{pre}0.33e6"
                elif DS == 8:
                    key = f"{pre}0.25e6"
                if key not in result:
                    result[key] = []
                result[key].append([snr, successful])

            # Write to JSON file
            with open("results/per_vs_fs.json", "w") as f:
                json5.dump(result, f)

    else:
        print("Nothing to be done. Exit")
        exit(0)
