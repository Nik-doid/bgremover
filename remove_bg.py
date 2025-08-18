import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import os
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
from io import BytesIO

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

from u2Net.model.u2net import U2NET


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


def _u2net_prob_mask(image_bytes: bytes, model_path: str) -> tuple[np.ndarray, tuple]:
    pil = Image.open(BytesIO(image_bytes)).convert('RGB')
    W, H = pil.size
    inp = _pil_to_tensor(pil)
    device = torch.device('cpu')

    model = U2NET(3, 1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        d1, *_ = model(inp)
        mask = d1[0][0].cpu().numpy()
        prob = _safe_normalize(mask)

    prob = cv2.resize((prob * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    prob = prob.astype(np.float32) / 255.0
    return prob, (W, H)


def _composite_on_white(image_bytes: bytes, alpha_mask_uint8: np.ndarray, transparent: bool = False) -> BytesIO:
    """Composite subject onto white or transparent background and return BytesIO."""
    original = Image.open(BytesIO(image_bytes)).convert('RGBA')

    if (original.size[0], original.size[1]) != (alpha_mask_uint8.shape[1], alpha_mask_uint8.shape[0]):
        alpha_mask_uint8 = cv2.resize(alpha_mask_uint8, (original.size[0], original.size[1]), interpolation=cv2.INTER_LINEAR)

    alpha_mask_uint8 = cv2.GaussianBlur(alpha_mask_uint8, (3, 3), 0)

    alpha = Image.fromarray(alpha_mask_uint8).convert('L')
    subject = original.copy()
    subject.putalpha(alpha)

    output_bytes = BytesIO()

    if transparent:
        subject.save(output_bytes, format="PNG")
    else:
        arr = np.array(subject)
        alpha_f = arr[..., 3:4] / 255.0
        arr[..., :3] = (arr[..., :3] * alpha_f + 255 * (1 - alpha_f)).astype(np.uint8)
        result = Image.fromarray(arr[..., :3], 'RGB')
        result.save(output_bytes, format="JPEG", quality=95)

    output_bytes.seek(0)
    return output_bytes

async def process_remove_background(file, transparent: bool, background_tasks: BackgroundTasks):
    """Handles FastAPI UploadFile -> runs background removal fully in memory -> returns StreamingResponse"""
    content = await file.read()
    filename_base = os.path.splitext(file.filename)[0]
    output_bytes = None
    used_rembg = False

    if REMBG_AVAILABLE:
        rgba = rembg_remove(content)
        cut = Image.open(BytesIO(rgba)).convert('RGBA')
        alpha = np.array(cut.split()[-1])
        if alpha.max() > 0:
            output_bytes = _composite_on_white(content, alpha, transparent=transparent)
            used_rembg = True

    if not used_rembg:
        prob, _ = _u2net_prob_mask(content, "saved_models/u2net/u2net_human_seg.pth")
        alpha = (prob * 255).astype(np.uint8)
        output_bytes = _composite_on_white(content, alpha, transparent=transparent)

    ext = "png" if transparent else "jpg"
    output_filename = f"{filename_base}.{ext}"

    return StreamingResponse(
        output_bytes,
        media_type="image/png" if transparent else "image/jpeg",
        headers={"Content-Disposition": f"inline; filename={output_filename}"}
    )