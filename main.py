import os
import time
import torch
import cv2
import argparse
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

torch.cuda.empty_cache()

def upscale_images(input_dir, output_dir, anime, scale_setting):
    """Апскейлит все изображения в папке с учётом ориентации и заданного масштаба"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Используется устройство: {device}")

    # Загрузка модели Real-ESRGAN
    print("📦 Загрузка модели Real-ESRGAN...")

    if not anime:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = "models\\RealESRGAN_x4plus.pth"
    else:
        # anime model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_path = "models\\RealESRGAN_x4plus_anime_6B.pth"


    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Найдено {len(images)} изображений. Начинаем обработку...")

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Ошибка чтения: {img_name}")
            continue

        h, w, _ = img.shape
        orientation = "portrait" if h > w else "landscape"

        # Определяем целевое разрешение по ориентации и scale-параметру
        if scale_setting == 2:
            target_height, target_width = (1920, 1080)
        elif scale_setting == 4:
            target_height, target_width = (2160, 3840)
        elif scale_setting == 8:
            target_height, target_width = (4320, 7680)
        elif scale_setting == 16:
            target_height, target_width = (8640, 15360)
        else:
            raise ValueError("Неверное значение scale! Используй 2, 4, 8 или 16")

        # Меняем местами, если изображение портретное
        if orientation == "portrait":
            target_height, target_width = target_width, target_height

        # Апскейл
        try:
            sr_image, _ = upsampler.enhance(img, outscale=scale_setting)
            sr_image = cv2.resize(sr_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        except RuntimeError as e:
            print(f"💥 Ошибка при обработке {img_name}: {e}")
            continue

        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, sr_image)
        torch.cuda.empty_cache()
        print(f"✅ Сохранено: {out_path}")

    print("🎉 Обработка завершена!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale изображений с Real-ESRGAN")
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с исходными изображениями")
    parser.add_argument("--output_dir", type=str, required=True, help="Папка для сохранения результата")
    parser.add_argument("--scale", type=int, default=4, help="Уровень upscale: 2, 4, 8, 16")
    parser.add_argument("--anime", action="store_true", help="Использовать модель для аниме")
    args = parser.parse_args()

    upscale_images(args.input_dir, args.output_dir, args.anime, args.scale)
