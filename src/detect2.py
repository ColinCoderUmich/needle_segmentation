import os
import glob
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from obtain_model import get_model

# ------------------
# Config
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "C:/Users/lamcolin/Documents/igmr/machine_learning/needle_segmentation/saved_models/UNeXt_s_0.pth"
image_dir  = "C:/Users/lamcolin/Documents/igmr/machine_learning/needle_segmentation/data/videos/data/label_frames"
output_dir = os.path.join(image_dir, "predictions")

os.makedirs(output_dir, exist_ok=True)

# ------------------
# Load model
# ------------------
model = get_model("UNeXt").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ------------------
# Image transform
# ------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # (H,W,C) → (C,H,W), [0,1]
    # add Normalize(...) ONLY if used in training
])

# ------------------
# Inference loop
# ------------------
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

print(f"Running inference on {len(image_paths)} images")

with torch.no_grad():
    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")  # or "L" if grayscale
        x = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)

        logits = model(x)
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()  # binary mask

        # save prediction
        out_path = os.path.join(output_dir, f"pred_{idx:06d}.png")
        torchvision.utils.save_image(preds, out_path)

print("Inference complete.")
