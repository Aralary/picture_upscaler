import os
import site

# Определяем путь к site-packages
try:
    site_packages_dirs = site.getsitepackages()  # системные site-packages
except AttributeError:
    # В некоторых окружениях (например, venv) getsitepackages может не работать
    import sysconfig
    site_packages_dirs = [sysconfig.get_paths()["purelib"]]

found = False
for sp in site_packages_dirs:
    target_file = os.path.join(sp, "basicsr", "data", "degradations.py")
    if os.path.exists(target_file):
        found = True
        break

if not found:
    print(f"❌ Файл degradations.py не найден в site-packages: {site_packages_dirs}")
    exit(1)

# Читаем содержимое файла
with open(target_file, "r", encoding="utf-8") as f:
    code = f.read()

# Исправляем импорт, если нужно
if "functional_tensor" in code:
    code = code.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale"
    )
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"✅ Импорт успешно исправлен: {target_file}")
else:
    print(f"⚡️ Исправление не требуется — всё уже в порядке: {target_file}")
