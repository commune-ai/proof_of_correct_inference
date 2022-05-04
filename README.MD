# Experiments for the viability of simple input/output quantization for proof of correct inference

This repository is still very experimental, but the core functionality is defined in `run_onnx.py`, specifically the `quantize` function. This file also contains a few helper functions.

One test is defined in `vision_models.py`: This involves running images through a PyTorch model, converting the model to ONNX, and then running the same images through the ONNX version of the model. The outputs are quantized to a variable degree, and the hash of the data is computed.
- What is tricky in this situation, comparing the hashes directly, is that any deviations that are close to one of the quantization bin edges matter greatly. Even arbitrarily small deviations near the bin edge can cause the hash to differ.
- The conclusion of this experiment is that this can work, but you really need to have a notion of a primary output and a validating output. Take the difference of these, and quantize and hash that. That way, hashes differ only when there are real deviations between the two outputs, not when there are small deviations that just happen to be near a quantization bin edge.

The version with separate primary and validation outputs is defined in `run_primary_inference.py` and `run_onnx_validation.py`. The upshot is as decribed above, but really it's just saying that if you take the output from different systems and take the difference in the float/64-bit quantized outputs, that difference should be small (duh). But the point is that this should be sufficient for detecting if the result is close enough to be the same, vs. truely a different output.

