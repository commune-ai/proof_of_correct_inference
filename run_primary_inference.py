import torch
import hashlib
import sys
import onnxruntime
import onnx
import numpy as np
import glob
from tqdm import tqdm
import os
from skimage import io, transform
from torchvision import datasets, models, transforms
from google.cloud import storage

from run_onnx import image_dataset_local, load_images, load_onnx_model, quantize

image_dataset_path = 'gs://scalable-magic-art/other_datasets/small_image_val_set/ILSVRC2012_img_val/'
storage_client = storage.Client("spartan-vertex-254018")

def download_images():
    bucket_name = image_dataset_path.split('/')[2]
    bucket = storage_client.get_bucket(bucket_name)
    prefix = '/'.join(image_dataset_path.split('/')[3:])
    blobs = bucket.list_blobs(prefix=prefix)
    os.makedirs(image_dataset_local, exist_ok=True)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        blob.download_to_filename(os.path.join(image_dataset_local, filename))

@torch.no_grad()
def run_pytorch(images, model, device='cuda'):
    images_pt = torch.tensor(np.transpose(images, (0, 3, 1, 2))) # [N, C, H, W]
    images_pt = images_pt.float().to(device)
    
    output = model(images_pt).cpu()
    preds = output.argmax(dim=1)

    return output, preds

def convert_to_onnx(torch_model, batch_size=1, onnx_filename='models/tmp.onnx'):
    torch.manual_seed(123)
    model_input = torch.randn(batch_size, 3, 224, 224, requires_grad=False, device='cuda', dtype=torch.float)
    torch.onnx.export(
            torch_model,
            model_input,
            onnx_filename,
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=10,          # the ONNX version to export the model to
            do_constant_folding=False,  # whether to execute constant folding for optimization
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    return onnx_filename

if __name__ == '__main__':
    max_mag_bits = 6
    quantize_bits = 63
    n_images = 10
    model_name = 'vgg16'
    print('Loading images...')
    if not os.path.exists(image_dataset_local):
        download_images()

    images = load_images(image_dataset_local, n=n_images)
    images = images.astype(np.float32)

    print('Quantizing input...')
    images_uint64, images_float64_converted = quantize(images,
            quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    input_hash_string = hashlib.sha256(images_uint64.tobytes()).hexdigest()
    with open(f'image_hashes/{n_images}', 'w') as f:
        f.write(input_hash_string)

    print('Loading model...')
    model = getattr(models, model_name)(pretrained=True).cuda().eval()
    
    print('Running pytorch model')
    output, preds = run_pytorch(images, model, device='cuda')
    output = output.numpy()
    output_uint64, output_float64_converted = quantize(output,
            quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    np.savez_compressed(f'primary_inference_output/{model_name}_pt_N{n_images}_mmb{max_mag_bits}_qb{quantize_bits}.npz',
            array=output_uint64)

    print('Converting to onnx...')
    onnx_filename = f'models/{model_name}.onnx'
    convert_to_onnx(model, onnx_filename=onnx_filename)
    ort_session = load_onnx_model(onnx_filename)

    print('Running onnx model...')
    images_onnx = np.transpose(images, (0, 3, 1, 2))
    ort_inputs = {'input': images_onnx}
    ort_output = ort_session.run(None, ort_inputs)[0]
    ort_preds = np.argmax(ort_output, axis=1)
    print(ort_output)

    ort_output_uint64, ort_output_float64_converted = quantize(ort_output,
            quantize_bits=quantize_bits, max_mag_bits=max_mag_bits)
    np.savez_compressed(f'primary_inference_output/{model_name}_onnx_N{n_images}_mmb{max_mag_bits}_qb{quantize_bits}.npz',
            array=ort_output_uint64)
