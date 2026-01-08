import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [2, 0], "Requires PyTorch >= 2.0"
print("PyTorch Version:", torch.__version__)

import time
def timed(fn, n_test: int, dtype: torch.dtype) -> tuple:
    """
    Measure the execution time for a given function.

    Args:
    - fn (function): The function to be timed.
    - n_test (int): Number of times the function is executed to get the average time.
    - dtype (torch.dtype): Data type for PyTorch tensors.

    Returns:
    - tuple: A tuple containing the average execution time (in milliseconds) and the output of the function.
    """
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        dt_loop_sum = []
        for _ in range(n_test):
            torch.cuda.synchronize()
            start = time.time()
            output = fn()
            torch.cuda.synchronize()
            end = time.time()
            dt_loop_sum.append(end-start)
        dt_test = sum(dt_loop_sum) / len(dt_loop_sum)

    return dt_test * 1000, output

#Accelerate ResNet-152 with torch.compile

# Download an example image from the pytorch website
import urllib
import matplotlib.pyplot as plt
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

from PIL import Image
input_image = Image.open(filename)
plt.imshow(input_image)
plt.axis('off')
plt.show()

import torch
import torchvision.transforms as transforms

# create the image preprocessor
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# load the resnet152 model
model = torch.hub.load('pytorch/vision:v0.17.2', 'resnet152', pretrained=True)
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output.shape)

def print_topk_labels(output, k):
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    topk_prob, topk_catid = torch.topk(probabilities, k)
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())

print_topk_labels(output, 5)

n_warmup = 10
n_test = 20
dtype = torch.bfloat16
inference_time=[]
mode=[]

t_warmup, _ = timed(lambda:model(input_batch), n_warmup, dtype)
t_test, output = timed(lambda:model(input_batch), n_test, dtype)
print(f"Average inference time for resnet152(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for resnet152(test): dt_test={t_test} ms")
print_topk_labels(output, 5)
inference_time.append(t_test)
mode.append("eager")

#Performance Evaluation of ResNet-152 Model in torch.compile(default) Mode
#clean up the workspace with torch._dynamo.reset().
torch._dynamo.reset()
model_opt1 = torch.compile(model, fullgraph=True)
t_compilation, _ = timed(lambda:model_opt1(input_batch), 1, dtype)
t_warmup, _ = timed(lambda:model_opt1(input_batch), n_warmup, dtype)
t_test, output = timed(lambda:model_opt1(input_batch), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for compiled resnet152(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for compiled resnet152(test): dt_test={t_test} ms")
print_topk_labels(output, 5)
inference_time.append(t_test)
mode.append("default")

#Performance Evaluation of ResNet-152 Model in torch.compile(reduce-overhead) Mode
torch._dynamo.reset()
model_opt2 = torch.compile(model, mode="reduce-overhead", fullgraph=True)
t_compilation, _ = timed(lambda:model_opt2(input_batch), 1, dtype)
t_warmup, _ = timed(lambda:model_opt2(input_batch), n_warmup, dtype)
t_test, output = timed(lambda:model_opt2(input_batch), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for compiled resnet152(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for compiled resnet152(test): dt_test={t_test} ms")
print_topk_labels(output, 5)
inference_time.append(t_test)
mode.append("reduce-overhead")

#Performance Evaluation of ResNet-152 Model in torch.compile(max-autotune) Mode
torch._dynamo.reset()
model_opt3 = torch.compile(model, mode="max-autotune", fullgraph=True)
t_compilation, _ = timed(lambda:model_opt3(input_batch), 1, dtype)
t_warmup, _ = timed(lambda:model_opt3(input_batch), n_warmup, dtype)
t_test, output = timed(lambda:model_opt3(input_batch), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for compiled resnet152(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for compiled resnet152(test): dt_test={t_test} ms")
print_topk_labels(output, 5)
inference_time.append(t_test)
mode.append("max-autotune")

import matplotlib.pyplot as plt

# Plotting the bar graph
plt.bar(mode, inference_time)
print(inference_time)
print(mode)

# Adding labels and title
plt.xlabel('mode')
plt.ylabel('Inference time (ms)')
plt.title('ResNet-152')

# Displaying the plot
plt.show()