import torch


def detect_vram():
    """Определяет объём VRAM (в гигабайтах)."""
    if not torch.cuda.is_available():
        print("⚠️ CUDA не найдена, используется CPU.")
        return 0
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"💾 VRAM: {vram_gb:.2f} GB")
    return vram_gb

def auto_tile(vram_gb):
    """Подбирает оптимальный tile под размер VRAM."""
    if vram_gb == 0:
        return 128
    elif vram_gb < 6:
        return 128
    elif vram_gb < 10:
        return 256
    elif vram_gb < 16:
        return 512
    elif vram_gb < 24:
        return 768
    else:
        return 1024
    