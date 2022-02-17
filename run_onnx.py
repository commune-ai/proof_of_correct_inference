import hashlib
import sys
import onnxruntime
import numpy as np
import glob
import os
from skimage import io, transform

image_dataset_local = 'cache/ILSVRC2012_img_val/'
def load_images(image_dataset_local, n=None, start=0):
    filenames = list(sorted(glob.glob(f'{image_dataset_local}/*.JPEG')))

    if n is not None:
        filenames = filenames[start:n+start]

    imgs = []
    for filename in filenames:
        img = io.imread(filename)
        img = img.astype(float)/255

        # crop
        img_size = 256
        img = transform.resize(img, (img_size, img_size, 3), order=3,
                anti_aliasing=True, preserve_range=True)

        # resize
        center_crop = 224
        crop_side = (img_size-center_crop)//2
        img = img[crop_side:-crop_side, crop_side:-crop_side]

        # normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean)/std
        imgs.append(img)

    return np.array(imgs).astype(np.float32)

def load_onnx_model(onnx_filename='tmp.onnx', gpu=False):
    if gpu:
        providers=['CUDAExecutionProvider']
    else:
        providers=['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_filename, providers=providers)
    return ort_session

def quantize(outputs, quantize_bits=32, max_mag_bits=6):
    # first convert to 64 bit int

    # asserts outputs are all between [-2**(max_mag_bits-1), 2**(max_mag_bits-1))
    max_mag = 2**(max_mag_bits-1)
    min_mag = -2**(max_mag_bits-1)

    if np.any(outputs >= max_mag):
        raise ValueError(f'outputs have magnitude >= {max_mag}. Max magnitude is {np.max(outputs)}')

    if np.any(outputs < min_mag):
        raise ValueError(f'outputs have magnitude < {min_mag}. Min magnitude is {np.min(outputs)}')

    outputs_float64 = outputs.astype(np.float64)
    outputs_float64_shifted = outputs_float64 - min_mag
    # max of outputs_float64_shifted is now 2**max_mag_bits
    # 2**max_mag_bits should get mapped to 2**quantize_bits
    # 0 should get mapped to 0
    outputs_float64_scaled = outputs_float64_shifted*(2**(quantize_bits-max_mag_bits))
    # outputs_float64_scaled is between [0, 2**quantize_bits)
    outputs_uint64 = outputs_float64_scaled.astype(np.uint64)

    outputs_float64_converted_scaled = outputs_uint64.astype(np.float64)
    outputs_float64_converted_shifted = outputs_float64_converted_scaled/(2**(quantize_bits-max_mag_bits))
    outputs_float64_converted = outputs_float64_converted_shifted + min_mag
    return outputs_uint64, outputs_float64_converted

def get_step_size(quantize_bits=32, max_mag_bits=6):
    # step size is 1/2**quantize_bits = ss/2**max_mag_bits
    return 2**(max_mag_bits-quantize_bits)

if __name__ == '__main__':
    quantize_bits = 11
    n_images = 1
    start = 74
    nbytes = 1
    startbytes = 1464
    print('Loading images...')
    images = load_images(image_dataset_local, n=n_images, start=start)

    print('Quantizing input...')
    images_uint64, images_float64_converted = quantize(images, quantize_bits=quantize_bits)
    print(hashlib.sha256(images_uint64.tobytes()).hexdigest())

    print('Loading onnx model...')
    ort_session = load_onnx_model()

    print('Running onnx model...')
    images_onnx = np.transpose(images, (0, 3, 1, 2))
    ort_inputs = {'input': images_onnx}
    ort_output = ort_session.run(None, ort_inputs)[0]
    ort_preds = np.argmax(ort_output, axis=1)

    ort_output_uint64, ort_output_float64_converted = quantize(ort_output, quantize_bits=quantize_bits)
    
    print(ort_output)
    print(ort_output_float64_converted)
    print(ort_output[0, 183], ort_output_uint64[0, 183], ort_output_float64_converted[0, 183])
    print(hashlib.sha256(ort_output_uint64.tobytes()[startbytes:startbytes+nbytes]).hexdigest())
