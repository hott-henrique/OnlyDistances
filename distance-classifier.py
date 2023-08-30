import argparse, collections, os, sys
import typing as t

import numpy as np
import tqdm

from scipy.spatial.distance import cosine, euclidean

class Classifier:

    def __init__(self, distance_func: t.Callable[[np.ndarray, np.ndarray], t.SupportsFloat]):
        self.observations = collections.defaultdict(list)
        self.distance_func = distance_func

    def observe(self, label: int, img: np.ndarray):
        self.observations[label].append(img)

    def predict(self, in_img: np.ndarray) -> int:
        d_min, label_min = np.inf, int()

        for label, images in self.observations.items():
            for img in images:
                d = np.float64(self.distance_func(img, in_img))

                if d < d_min:
                    d_min = d
                    label_min = label

        return label_min

def read_image(If: t.BinaryIO) -> tuple[int, np.ndarray]:
    label = int.from_bytes(If.read(1), byteorder="big")
    pixels = np.frombuffer(If.read(1024 * 3), dtype=np.uint8)
    return label, pixels

def main(cifar10_path: str, img_type: str, distance_metric: str):
    if not os.path.exists(cifar10_path):
        print(f"Could not find CIFAR10 dataset in path: {cifar10_path}", file=sys.stderr)
        return

    dfunc: t.Callable[[np.ndarray, np.ndarray], t.SupportsFloat]
    match distance_metric:
        case "manhattan":
            dfunc = lambda u, v: np.float64(np.sum(np.abs(u - v)))
        case "cosine":
            dfunc = cosine
        case "euclidian":
            dfunc = euclidean

    agent = Classifier(distance_func=dfunc)

    for i in tqdm.tqdm(range(5), desc="Observing", leave=False):
        input_file_name = os.path.join(cifar10_path, f"{img_type}_data_batch_{i + 1}.bin")
        with open(file=input_file_name, mode="rb") as If:
            for _ in range(10000):
                label, pixels = read_image(If=If)
                agent.observe(label=label, img=pixels)

    count_errors, N_tests = 0, 10000

    input_file_name = os.path.join(cifar10_path, f"{img_type}_test_batch.bin")
    with open(file=input_file_name, mode="rb") as If:
        for _ in tqdm.tqdm(range(N_tests), desc="Testing", leave=True):
            label, pixels = read_image(If=If)
            predicted_label = agent.predict(pixels)

            if label != predicted_label:
                count_errors = count_errors + 1

    print(f"Result: {N_tests - count_errors}/{N_tests}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cifar10-path", type=str, required=True)
    parser.add_argument("--img-type", choices=[ "rgb", "gs" ], default="rgb")
    parser.add_argument("--distance-metric", choices=[ "manhattan", "euclidian", "cosine" ], required=True)

    args = parser.parse_args()

    main(cifar10_path=args.cifar10_path,
         img_type=args.img_type,
         distance_metric=args.distance_metric)

