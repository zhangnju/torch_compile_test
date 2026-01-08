import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import matplotlib.pyplot as plt

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


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()

# load the image processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")

if torch.cuda.is_available():
    inputs = inputs.to('cuda')
    model.to('cuda')
    
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

n_warmup = 10
n_test = 20
dtype = torch.bfloat16
inference_time=[]
mode=[]

#Performance Evaluation of Vision Transformer Model in Eager Mode
torch._dynamo.reset()
t_warmup, _ = timed(lambda:model(**inputs), n_warmup, dtype)
t_test, output = timed(lambda:model(**inputs), n_test, dtype)
print(f"Average inference time for ViT(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for ViT(test): dt_test={t_test} ms")
inference_time.append(t_test)
mode.append("eager")
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = output.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

#Performance Evaluation of Vision Transformer Model in torch.compile(default) Mode
torch._dynamo.reset()
model_opt1 = torch.compile(model, fullgraph=True)
t_compilation, _ = timed(lambda:model_opt1(**inputs), 1, dtype)
t_warmup, _ = timed(lambda:model_opt1(**inputs), n_warmup, dtype)
t_test, output = timed(lambda:model_opt1(**inputs), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for ViT(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for ViT(test): dt_test={t_test} ms")
inference_time.append(t_test)
mode.append("default")
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = output.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

#Performance Evaluation of Vision Transformer Model in torch.compile(reduce-overhead) Mode
torch._dynamo.reset()
model_opt2 = torch.compile(model, mode="reduce-overhead", fullgraph=True)
t_compilation, _ = timed(lambda:model_opt2(**inputs), 1, dtype)
t_warmup, _ = timed(lambda:model_opt2(**inputs), n_warmup, dtype)
t_test, output = timed(lambda:model_opt2(**inputs), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for ViT(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for ViT(test): dt_test={t_test} ms")
inference_time.append(t_test)
mode.append("reduce-overhead")
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = output.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

#Performance Evaluation of Vision Transformer Model in torch.compile(max-autotune) Mode
torch._dynamo.reset()
model_opt3 = torch.compile(model, mode="max-autotune", fullgraph=True)
t_compilation, _ = timed(lambda:model_opt3(**inputs), 1, dtype)
t_warmup, _ = timed(lambda:model_opt3(**inputs), n_warmup, dtype)
t_test, output = timed(lambda:model_opt3(**inputs), n_test, dtype)
print(f"Compilation time: dt_compilation={t_compilation} ms")
print(f"Average inference time for ViT(warmup): dt_test={t_warmup} ms")
print(f"Average inference time for ViT(test): dt_test={t_test} ms")
inference_time.append(t_test)
mode.append("max-autotune")
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = output.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

# Plotting the bar graph
plt.bar(mode, inference_time)
print(inference_time)
print(mode)

# Adding labels and title
plt.xlabel('mode')
plt.ylabel('Inference time (ms)')
plt.title('ViT')

# Displaying the plot
plt.show()