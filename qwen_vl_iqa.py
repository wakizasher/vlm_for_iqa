import os
import csv
import time
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


tic = time.perf_counter()
# Load the model and processor
model_id = "Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use lower precision for 16GB VRAM
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)


# Define the folder containing images and output CSV file
image_folder = r"D:\iNaturalist\test_3000"


output_csv = "blurry_results_qwen.csv"
prompt = f"Does the image contain other taxa than the one? Answer only Yes or No."


# Check if the directory exists
if not os.path.exists(image_folder):
    print(f"Error: Directory {image_folder} does not exist. Please create it and add your images.")
    exit(1)


# Prepare CSV file to store results
results = []
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
              if img.endswith(('.jpg', '.png','.JPG','.png','.PNG', '.jpeg'))]


if not image_paths:
    print(f"No images found in {image_folder}. Please add images in JPG, JPEG, or PNG format.")
    exit(1)


# Process each image
for image_file in image_paths:
    image_path = os.path.join(image_folder, image_file)
    try:
        # Prepare input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0].strip()

        # Extract Yes/No from response
        result = "Yes" if "Yes" in response else "No"
        results.append([image_file, result])
        print(f"Processed {image_file}: {result}")
    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")
        results.append([image_file, "Error"])


# Save results to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Blurry"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")


toc = time.perf_counter()
print(f"Model finished in {toc - tic:0.4f}")