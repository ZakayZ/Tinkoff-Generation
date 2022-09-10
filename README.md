# Решение шестого задания отбора в Tinkoff Generation

В этом репозитории приведена как N-граммная модель, так и нейронная сеть.

## Руководство пользователя

### Для обучения любой модели нужно запустить `train.py`. 

В качестве аргументов скрипта требуется ввести:

 - --input_dir - папка с тренировочными данными в формате `.txt`.
 - --model - путь до файла(без расширения), в который модель будет сохранена по окончанию обучения.

Также можно указать:

 - --gram_length - длина префикса, который будет использовать модель.
 - --model_type - логическая перменная, которая показывает обучать нейронка(1) или же обычную модель(0).
 - --epochs - в случае обучения нейронной сети указывает количество эпох.
 - --learning_rate - в случае обучения нейронной сети указывает начальный lr.

### Для создания текста нужно запустить `generate.py`.

В качестве параметров указать:

 - --model - путь до файла(без расширения) с моделью, которую мы хотим использовать.
 - --length - количество слов, которое мы хотим сгенерировать.

Также можно указать:

 - --prefix - начало текста.
 - --model_type - если выбрали файл с нейронной сетью, то необходимо указать здесь 1.

## Примечания

1. Кроме требуемых файлов присутвует 2 файла с кодом для моделей, так как их нужно было бы дублировать в `train.py` и 
`generate.py`

2. Приведено 2 обученные модели `ml_model.pkl`, которая использует N-граммную структуру, и `nn_model.pkl`, которая преобразует эмбеддинги последних N слов для генерации нового слова. Обе обучены на серии книг "Война и мир", причем нейронка обучалась меньше часа.

3. Приоритетным для поступления для меня будет DL.