import torch
import torch_tensorrt

# Packages to make TensorRT work
# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install nvidia-pyindex
# pip install nvidia-tensorrt
# pip install torch-tensorrt==1.2.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.2.0
# might be necessary to add LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=~/.virtualenvs/test/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/


file_name = "defects_pytorch_Unet_se_resnext50_32x4d_c=2_aug=False_FocalLoss_iou-0.9860.pt"
best_model = torch.load(f"./weights/{file_name}")

best_model.eval()
# Trace model on cpu cause of GPU memory issues
traced_model = torch.jit.trace(best_model, torch.randn([1, 3, 768, 960], 
                                dtype=torch.float32))
DEVICE = 'cuda'
traced_model.to(DEVICE)
traced_model.eval()

# If fast fp16 operations are supported on GPU
# traced_model.half()
# trt_model = torch_tensorrt.compile(traced_model, 
#                                     inputs=[torch_tensorrt.Input([1, 3, 768, 960], dtype=torch.half)], 
#                                     enabled_precisions = {torch.half})

# If fast fp16 operations are not supported on GPU
trt_model = torch_tensorrt.compile(traced_model, 
                                    inputs=[torch_tensorrt.Input([1, 3, 768, 960], dtype=torch.float32)], 
                                    enabled_precisions = {torch.float32})

# Save as TorchScript
save_name = "defects_trt" + file_name[15: -2] + "ts"
torch.jit.save(trt_model, f"./weights/trt/{save_name}")
