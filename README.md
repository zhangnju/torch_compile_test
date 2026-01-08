# example codes to test torch.compile feature 

## official torch.compile
inductor is the default backend, and test and compare the inference time across various modes: Eager, default, reduce-overhead, and max-autotune. 

```sh
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

pip install -q transformers sentencepiece numpy tabulate scipy matplotlib sentencepiece huggingface_hub

python resnet.py

python vit.py

```

## Vendor version 
default:python3 iree_test.py
run test with iree: python3 iree_test.py --iree True 
