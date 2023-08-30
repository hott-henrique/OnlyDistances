source .venv/bin/activate

echo "Invoking: Manhattan"

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type rgb \
    --distance-metric manhattan > .data/rgb-manhattan.stdout 2> .data/rgb-manhattan.stderr &

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type gs \
    --distance-metric manhattan > .data/gs-manhattan.stdout 2> .data/gs-manhattan.stderr &

wait $!

echo "Invoking: Euclidian"

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type rgb \
    --distance-metric euclidian > .data/rgb-euclidian.stdout 2> .data/rgb-euclidian.stderr &

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type gs \
    --distance-metric euclidian > .data/gs-euclidian.stdout 2> .data/gs-euclidian.stderr &

wait $!

echo "Invoking: Cosine"

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type rgb \
    --distance-metric cosine > .data/rgb-cosine.stdout 2> .data/rgb-cosine.stderr &

python3 distance-classifier.py \
    --cifar10-path .data/preprocess/cifar-10/ \
    --img-type gs \
    --distance-metric cosine > .data/gs-cosine.stdout 2> .data/rgb-cosine.stderr &

wait $!

echo "Finished all executions."

