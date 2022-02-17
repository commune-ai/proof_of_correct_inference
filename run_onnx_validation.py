import hashlib
import sys
import onnxruntime
import numpy as np

from run_onnx import image_dataset_local, load_images, load_onnx_model, quantize, get_step_size

if __name__ == '__main__':
    max_mag_bits = 6
    quantize_bits = 63
    n_images = 10
    model_name = 'vgg16'
    compare_against = 'pt'
    gpu = True
    print('Loading images...')
    images = load_images(image_dataset_local, n=n_images)
    print(images)

    print('Quantizing input...')
    images_uint64, images_float64_converted = quantize(images,
            quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    input_hash = hashlib.sha256(images_uint64.tobytes()).hexdigest()
    with open(f'image_hashes/{n_images}', 'r') as f:
        reported_input_hash = f.read().strip()

    print(f'Reported input hash: {reported_input_hash}')
    print(f'Calculated input hash: {input_hash}')

    print('Loading onnx model...')
    onnx_filename = f'models/{model_name}.onnx'
    ort_session = load_onnx_model(onnx_filename, gpu=True)

    print('Running onnx model...')
    images_onnx = np.transpose(images, (0, 3, 1, 2))
    ort_inputs = {'input': images_onnx}
    ort_output = ort_session.run(None, ort_inputs)[0]
    ort_preds = np.argmax(ort_output, axis=1)
    print(ort_output)

    ort_output_uint64, ort_output_float64_converted = quantize(ort_output,
            quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    
    print('Loading reported results...')
    reported_output_uint64 = np.load(f'primary_inference_output/{model_name}_{compare_against}_N{n_images}_mmb{max_mag_bits}_qb{quantize_bits}.npz')['array']
    diff_ort = ort_output_uint64.astype(np.int64) - reported_output_uint64.astype(np.int64)
    largest_deviation_int = np.max(np.abs(diff_ort))
    largest_deviation_float = largest_deviation_int*get_step_size(quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    print(largest_deviation_int)
    print(largest_deviation_float)
