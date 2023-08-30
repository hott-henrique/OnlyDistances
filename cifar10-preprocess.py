import os, sys
import typing as t

import numpy as np
import cv2 as ocv


def read_image_components(If: t.BinaryIO) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    label = int.from_bytes(If.read(1), byteorder="big")
    R = np.frombuffer(If.read(1024), dtype=np.uint8)
    G = np.frombuffer(If.read(1024), dtype=np.uint8)
    B = np.frombuffer(If.read(1024), dtype=np.uint8)
    return label, R, G, B

def components2image(components: tuple[np.ndarray, np.ndarray, np.ndarray], img: np.ndarray):
    R, G, B = components

    for i in range(32):
        base = i * 32
        final = base + 32

        # At row i, select all columns, and in
        # ... these columns select the nth attribute.
        img[i, :, 0] = R[base: final]
        img[i, :, 1] = G[base: final]
        img[i, :, 2] = B[base: final]

def read_image(If: t.BinaryIO, img: np.ndarray) -> int:
    label, R, G, B = read_image_components(If)
    components2image(components=(R, G, B), img=img)
    return label

def convert_file(input_file_name: str,
                 output_rgb_file_name: str,
                 output_gs_file_name: str,
                 img: np.ndarray,
                 gs_img: np.ndarray):
    with open(file=input_file_name, mode="rb") as If, \
         open(file=output_rgb_file_name, mode="wb") as rgbOf, \
         open(file=output_gs_file_name, mode="wb") as gsOf:
        # The description says that each file has 10000 images.
        for _ in range(10000):
            label = read_image(If=If, img=img)

            rgbOf.write(label.to_bytes())
            rgbOf.write(img.tobytes())

            ocv.cvtColor(img, ocv.COLOR_RGB2GRAY, dst=gs_img)

            gsOf.write(label.to_bytes())
            gsOf.write(gs_img.tobytes())

def main():
    CIFAR10_BASE_PATH = ".data/cifar-10-binary/cifar-10-batches-bin/"

    if not os.path.exists(CIFAR10_BASE_PATH):
        print(f"Could not find CIFAR10 dataset in path: {CIFAR10_BASE_PATH}", file=sys.stderr)
        return

    PREPROCESS_BASE_PATH = ".data/preprocess/cifar-10/"

    os.makedirs(PREPROCESS_BASE_PATH, exist_ok=True)

    img = np.ndarray(shape=(32, 32, 3), dtype=np.uint8)
    gs_img = np.ndarray(shape=(32, 32, 3), dtype=np.uint8)

    for i in range(5):
        input_file_name = os.path.join(CIFAR10_BASE_PATH, f"data_batch_{i + 1}.bin")
        output_rgb_file_name = os.path.join(PREPROCESS_BASE_PATH, f"rgb_data_batch_{i + 1}.bin")
        output_gs_file_name = os.path.join(PREPROCESS_BASE_PATH, f"gs_data_batch_{i + 1}.bin")

        convert_file(input_file_name, output_rgb_file_name, output_gs_file_name, img, gs_img)

    input_file_name = os.path.join(CIFAR10_BASE_PATH, f"test_batch.bin")
    output_rgb_file_name = os.path.join(PREPROCESS_BASE_PATH, f"rgb_test_batch.bin")
    output_gs_file_name = os.path.join(PREPROCESS_BASE_PATH, f"gs_test_batch.bin")

    convert_file(input_file_name, output_rgb_file_name, output_gs_file_name, img, gs_img)

    os.system(f"cp {os.path.join(CIFAR10_BASE_PATH, 'batches.meta.txt')} {os.path.join(PREPROCESS_BASE_PATH, 'meta.txt')}")

if __name__ == "__main__":
    main()

