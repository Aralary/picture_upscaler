import os
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from .utils import detect_vram, auto_tile


def load_upsampler(anime: bool, device: torch.device):
    """Загружает и настраивает Real-ESRGANer."""
    model_path = get_model_path(anime)
    if anime:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6, num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)

    vram_gb = detect_vram()
    tile = auto_tile(vram_gb)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
        device=device,
    )
    return upsampler


def get_model_path(anime: bool) -> str:
    """Возвращает путь до модели в cross-platform формате."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_name = "RealESRGAN_x4plus_anime_6B.pth" if anime else "RealESRGAN_x4plus.pth"
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Не найдена модель: {model_path}")
    return model_path
