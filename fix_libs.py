import os

venv_path = os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages")
target_file = os.path.join(venv_path, "basicsr", "data", "degradations.py")

if not os.path.exists(target_file):
    print("❌ Файл degradations.py не найден:", target_file)
    exit(1)

with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

if "functional_tensor" in code:
    code = code.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale"
    )
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ Импорт успешно исправлен:", target_file)
else:
    print("⚡️ Исправление не требуется — всё уже в порядке.")
