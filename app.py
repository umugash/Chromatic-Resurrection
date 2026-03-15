# app.py
from flask import Flask, render_template, request, send_file
from model.gan import Generator
import torch
from torch import nn
from torchvision.utils import save_image
from skimage import color
from PIL import Image
import numpy as np
import io
import os

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Load Generator model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
checkpoint_path = "model/checkpoint_epoch_20.pth"
G.load_state_dict(torch.load(checkpoint_path, map_location=device)['G_state_dict'])
G.eval()

# -------------------------------
# Helper function to colorize image
# -------------------------------
def colorize_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256), Image.LANCZOS)
    img_np = np.array(img)/255.0
    lab = color.rgb2lab(img_np).astype(np.float32)
    L = lab[:,:,0:1]/100.0
    L_tensor = torch.from_numpy(L.transpose((2,0,1))).unsqueeze(0).float().to(device)

    with torch.no_grad():
        fake_ab = G(L_tensor)
    
    fake_lab = torch.cat([L_tensor, fake_ab], dim=1)[0].permute(1,2,0).cpu().numpy()
    fake_lab[:,:,0] *= 100.0
    fake_lab[:,:,1:3] *= 128.0
    rgb = color.lab2rgb(fake_lab)
    rgb_img = Image.fromarray((rgb*255).astype(np.uint8))
    return rgb_img

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Colorize
        colorized_img = colorize_image(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, "color_" + file.filename)
        colorized_img.save(output_path)

        return render_template("index.html", input_image=file_path, output_image=output_path)

    return render_template("index.html", input_image=None, output_image=None)

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
