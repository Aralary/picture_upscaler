import os
import argparse
import cv2
import torch
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


torch.cuda.empty_cache()


def upscale_images(input_dir, output_dir, target_height=16000):
    """
    Апскейлит все портретные изображения (высота > ширина) из input_dir до высоты ~target_height,
    сохраняя пропорции, с использованием Real‑ESRGAN.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Используется устройство: {device}")
    os.makedirs(output_dir, exist_ok=True)

    print("📦 Загрузка модели Real‑ESRGAN...")
    # Здесь укажи рабочий URL к весам или локальный путь:
    model_path = r".\\models\\RealESRGAN_x4plus_anime_6B.pth"


    # Для anime 6B (портреты)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=16, # чем больше тем быстрее работает
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )

    supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_ext)]
    if not files:
        print("❌ Нет поддерживаемых изображений в папке.")
        return

    print(f"🚀 Найдено {len(files)} изображений. Начинаем обработку...")
    for filename in tqdm(files, desc="Upscaling"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Не удалось открыть файл: {filename}")
            continue

        h, w = img.shape[:2]
        if h <= w:
            print(f"⚠️ Пропущено (не портрет): {filename}")
            continue

        sr_image, _ = upsampler.enhance(img, outscale=4.0)

        h_sr, w_sr = sr_image.shape[:2]
        scale_factor = target_height / h_sr
        new_w = int(w_sr * scale_factor)
        output_image = cv2.resize(sr_image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_path, output_image)
        torch.cuda.empty_cache()


    print("\n🎉 Обработка завершена! Результаты в:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upscale portrait images to ~16K height using Real‑ESRGAN"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с исходными изображениями")
    parser.add_argument("--output_dir", type=str, required=True, help="Папка для сохранения результатов")
    args = parser.parse_args()

    upscale_images(args.input_dir, args.output_dir)
