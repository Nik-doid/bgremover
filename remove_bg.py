import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

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


def _u2net_prob_mask(image_path: str, model_path: str) -> tuple[np.ndarray, tuple]:
    pil = Image.open(image_path).convert('RGB')
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


def _composite_on_white(image_path: str, alpha_mask_uint8: np.ndarray, out_path: str, transparent: bool = False):
    """
    Composite subject onto white (halo-free) or output transparent background.
    """
    original = Image.open(image_path).convert('RGBA')

    if (original.size[0], original.size[1]) != (alpha_mask_uint8.shape[1], alpha_mask_uint8.shape[0]):
        alpha_mask_uint8 = cv2.resize(alpha_mask_uint8, (original.size[0], original.size[1]), interpolation=cv2.INTER_LINEAR)

    alpha_mask_uint8 = cv2.GaussianBlur(alpha_mask_uint8, (3, 3), 0)

    alpha = Image.fromarray(alpha_mask_uint8).convert('L')
    subject = original.copy()
    subject.putalpha(alpha)

    if transparent:
        subject.save(out_path, format="PNG")
    else:
        arr = np.array(subject)
        alpha_f = arr[..., 3:4] / 255.0
        arr[..., :3] = (arr[..., :3] * alpha_f + 255 * (1 - alpha_f)).astype(np.uint8)
        result = Image.fromarray(arr[..., :3], 'RGB')
        result.save(out_path, quality=95)


def remove_background_strict(
    image_path: str,
    output_path: str,
    model_path: str = 'saved_models/u2net/u2net_human_seg.pth',
    use_rembg_first: bool = True,
    transparent: bool = False
):
    """Removes background and saves output with halo-free edges."""
    used_rembg = False

    if use_rembg_first and REMBG_AVAILABLE:
        with open(image_path, 'rb') as f:
            rgba = rembg_remove(f.read())
        from io import BytesIO
        cut = Image.open(BytesIO(rgba)).convert('RGBA')
        alpha = np.array(cut.split()[-1])
        if alpha.max() > 0:  
            _composite_on_white(image_path, alpha, output_path, transparent=transparent)
            used_rembg = True

    if not used_rembg:
        prob, _ = _u2net_prob_mask(image_path, model_path)
        alpha = (prob * 255).astype(np.uint8)
        _composite_on_white(image_path, alpha, output_path, transparent=transparent)

    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    input_image = "a.jpg" 
    output_image = "output.jpg" 

    remove_background_strict(
        input_image,
        output_image,
        model_path='saved_models/u2net/u2net_human_seg.pth',
        use_rembg_first=True,
        transparent=False
    )
