import argparse
from upscale.core import upscale_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale изображений с Real-ESRGAN")
    parser.add_argument("--input_dir", type=str, required=True, help="Папка с исходными изображениями")
    parser.add_argument("--output_dir", type=str, required=True, help="Папка для сохранения результата")
    parser.add_argument("--scale", type=int, default=4, help="Уровень upscale: 2, 4, 8, 16")
    parser.add_argument("--anime", action="store_true", help="Использовать модель для аниме")
    args = parser.parse_args()

    upscale_images(args.input_dir, args.output_dir, args.anime, args.scale)
