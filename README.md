# picture_upscaler
simple application for picture upscaling

Модели взяты с сайта:
https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md


# Usage

```
python -m venv venv
```
Aктивируем виртуальное окружение
```
source venv/bin/activate # for linux

.\venv\Scripts\activate # for windows
```

Устанавливаем все зависимости:
```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Запускаем
```
python main.py --input_dir {input} --output_dir {output}
```

- input путь до директории с портретами
- output путь до директории, в которую будут сохраняться результаты