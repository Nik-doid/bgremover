import os
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

from u2Net.model.u2net import U2NET

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models/u2net/u2net_human_seg.pth")

print("BASE_DIR:", BASE_DIR)
print("Checking U2NET model exists at:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"U2NET model file not found: {MODEL_PATH}")

device = torch.device("cpu")
u2net_model = U2NET(3, 1)
state = torch.load(MODEL_PATH, map_location=device)
u2net_model.load_state_dict(state)
u2net_model.eval()
print("U2NET model loaded successfully.")

def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

def _pil_to_tensor(pil_img: Image.Image, size=(320, 320)) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return tfm(pil_img).unsqueeze(0)

def _u2net_prob_mask(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(BytesIO(image_bytes)).convert('RGB')
    W, H = pil.size
    inp = _pil_to_tensor(pil)

    with torch.no_grad():
        d1, *_ = u2net_model(inp)
        mask = d1[0][0].cpu().numpy()
        prob = _safe_normalize(mask)

    prob = cv2.resize((prob * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    prob = prob.astype(np.float32) / 255.0
    return prob

def _composite_on_white(image_bytes: bytes, alpha_mask_uint8: np.ndarray) -> BytesIO:
    original = Image.open(BytesIO(image_bytes)).convert('RGBA')
    if (original.size[0], original.size[1]) != (alpha_mask_uint8.shape[1], alpha_mask_uint8.shape[0]):
        alpha_mask_uint8 = cv2.resize(alpha_mask_uint8, (original.size[0], original.size[1]), interpolation=cv2.INTER_LINEAR)
    alpha_mask_uint8 = cv2.GaussianBlur(alpha_mask_uint8, (3, 3), 0)
    alpha = Image.fromarray(alpha_mask_uint8).convert('L')
    subject = original.copy()
    subject.putalpha(alpha)

    arr = np.array(subject)
    alpha_f = arr[..., 3:4] / 255.0
    arr[..., :3] = (arr[..., :3] * alpha_f + 255 * (1 - alpha_f)).astype(np.uint8)
    result = Image.fromarray(arr[..., :3], 'RGB')

    output_bytes = BytesIO()
    result.save(output_bytes, format="JPEG", quality=95)
    output_bytes.seek(0)
    return output_bytes

@app.get("/", response_class=HTMLResponse)
async def form_page():
    return """
    <html>
        <head><title>Background Removal</title></head>
        <body style="font-family: sans-serif;">
            <h2>Upload an image to remove background</h2>
            <form action="/remove-bg" method="post" enctype="multipart/form-data">
                <label>Select Image:</label>
                <input type="file" name="file" accept="image/*" required><br><br>
                <button type="submit">Upload & Process</button>
            </form>
        </body>
    </html>
    """

@app.post("/remove-bg")
async def remove_bg(file: UploadFile, background_tasks: BackgroundTasks):
    content = await file.read()
    filename_base = os.path.splitext(file.filename)[0]
    output_bytes = None
    used_rembg = False

    print("Received file:", file.filename, len(content), "bytes")

    try:
        # --- Try RemBG first ---
        if REMBG_AVAILABLE:
            print("Using RemBG for background removal...")
            rgba = rembg_remove(content)
            cut = Image.open(BytesIO(rgba)).convert('RGBA')
            alpha = np.array(cut.split()[-1])
            if alpha.max() > 0:
                output_bytes = _composite_on_white(content, alpha)
                used_rembg = True

        # --- Fallback to U2NET ---
        if not used_rembg:
            print("Using U2NET for background removal...")
            prob = _u2net_prob_mask(content)
            alpha = (prob * 255).astype(np.uint8)
            output_bytes = _composite_on_white(content, alpha)

        output_filename = f"{filename_base}.jpg"
        print("Background removal successful:", output_filename)
        return StreamingResponse(
            output_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={output_filename}"}
        )
    except Exception as e:
        print("Error processing file:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
