# picture_upscaler
simple application for picture upscaling

Модели взяты с сайта:
https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md


# Использование
## Запуск без Docker
### Создаем виртуальное окружение

```
python -m venv venv
```

### Aктивируем виртуальное окружение
```
source venv/bin/activate # for linux

.\venv\Scripts\activate # for windows
```

### Устанавливаем все зависимости:
```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

### Исправляем небольшие проблемы:
```
python fix_libs.py
```

### Запускаем
```
python main.py --input_dir {input} --output_dir {output} --scale {numX} --anime
```

- флаг --input_dir путь до директории с портретами
- флаг --output_dir путь до директории, в которую будут сохраняться результаты
- флаг --scale задает до какого разрешения надо сделать upscale. Может принимать следующие значения: 2 , 4 , 8 , 16 которые означают upscale соответственно до (2к, 4к, 8к, 16к)
- флаг --anime лучше ставить если обрабатываете аниме картинки

## Запуск c помощью Docker

### Сборка
```
docker build -t picture-upscaler .
```

### Запуск (использует GPU и локальные папки)
```
docker run --gpus all -it --rm \
    -v "$(pwd)/input:/app/input" \
    -v "$(pwd)/output:/app/output" \
    picture-upscaler \
    --input_dir input \
    --output_dir output \
    --scale 8 \
    --anime
```