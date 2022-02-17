import numpy as np

from run_onnx import quantize, get_step_size

if __name__ == '__main__':
    a = np.array([-2, 1.99999]).astype(np.float32)
    a_uint64, a_float64_converted = quantize(a, quantize_bits=8, max_mag_bits=2)
    print(a)
    print(a_uint64)
    print(a_float64_converted)
    print(f'step size: {get_step_size(quantize_bits=8, max_mag_bits=2)}')
