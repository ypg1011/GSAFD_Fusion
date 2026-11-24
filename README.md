# Gradien-Based Stochasti Adaptiv Fourie Decompositio fo Multi Modal Imag Fusion
Gradient-Based Stochastic Adaptive Fourier Decomposition for Multi-Modal Image Fusion

This release provides inference code for aligned infrared–visible image pairs. Due to academic and research purposes, we will release the training code later.

## Environment

- Python 3.12
- PyTorch 2.5.0.
- NumPy
- OpenCV

In `args.py`, set `Args.cuda` to `1` for CUDA or `0` for CPU.

## Test

Place GSAFD checkpoint at `./model/afd_net.model` and set these values in `testing_fusion.py`.

- `model_path = "./model/afd_net.model"` for the checkpoint.
- `test_path = "./test/ir/"` for infrared inputs. Put the matching visible images in `./test/vis/`. Use the same filenames, extensions, and dimensions in both folders. With the current pairing code, reserve `ir` for the input folder name.
- `output_path_root = "outputs/M3FD/"` for fused PNGs.
