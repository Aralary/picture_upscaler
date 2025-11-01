import os
import cv2
import torch
from tqdm import tqdm
from .models import load_upsampler


def upscale_images(input_dir, output_dir, anime, scale_setting):
    """Апскейлит все изображения в папке."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Используется устройство: {device}")

    upsampler = load_upsampler(anime, device)
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("⚠️ Не найдено изображений для обработки.")
        return

    print(f"🚀 Найдено {len(images)} изображений. Начинаем обработку...")

    # Целевые размеры (можно потом вынести в конфиг)
    resolutions = {
        2: (1920, 1080),
        4: (2160, 3840),
        8: (4320, 7680),
        16: (8640, 15360)
    }

    if scale_setting not in resolutions:
        raise ValueError("❌ Неверный параметр --scale (2, 4, 8, 16).")
        
    for i, img_name in enumerate(tqdm(images, ncols=80)):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Ошибка чтения: {img_name}")
            continue

        h, w, _ = img.shape
        orientation = "portrait" if h > w else "landscape"


        target_height, target_width = resolutions[scale_setting]

        # Меняем местами для портретных изображений
        if orientation == "portrait":
            target_height, target_width = target_width, target_height

        try:
            sr_image, _ = upsampler.enhance(img, outscale=scale_setting)
            sr_image = cv2.resize(
                sr_image, (target_width, target_height),
                interpolation=cv2.INTER_CUBIC
            )

            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, sr_image)
            print(f"✅ Сохранено: {out_path}")

        except RuntimeError as e:
            print(f"💥 Ошибка при обработке {img_name}: {e}")
            torch.cuda.empty_cache()
            continue
        
        if i % 5 == 0:
            torch.cuda.empty_cache()

    print("🎉 Обработка завершена!")
