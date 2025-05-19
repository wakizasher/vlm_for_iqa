import os
import csv
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)


processor = AutoProcessor.from_pretrained("Qwen2.5-VL-3B-Instruct")


print("Model and processor loaded successfully")