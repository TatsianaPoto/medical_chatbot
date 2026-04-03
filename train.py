"""Обучение модели классификации намерений"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from config import config
from logger import logger
import os

def train_model():
    """Основная функция обучения модели"""

    logger.info("Начало обучения модели...")

    # Загрузка данных
    with open(config.INTENTS_FILE, 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    # Подготовка данных
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            # Токенизация
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Стемминг и удаление знаков препинания
    ignore_words = ['?', '!', '.', ',', ':', ';', '-', '_']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    logger.info(f"Размер словаря: {len(all_words)} слов")
    logger.info(f"Количество тегов: {len(tags)}")

    # Создание обучающих данных
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Параметры модели
    input_size = len(X_train[0])
    hidden_size = config.HIDDEN_SIZE
    output_size = len(tags)

    logger.info(f"Параметры модели: input={input_size}, hidden={hidden_size}, output={output_size}")

    # Dataset и DataLoader
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Устройство (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")

    # Создание модели
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Обучение
    logger.info("Начало обучения...")
    for epoch in range(config.NUM_EPOCHS):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # Прямое распространение
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Логирование каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            logger.info(f"Эпоха [{epoch+1}/{config.NUM_EPOCHS}], Loss: {loss.item():.4f}")

    logger.info(f"Обучение завершено. Финальная loss: {loss.item():.4f}")

    # Сохранение модели
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    torch.save(data, config.MODEL_PATH)
    logger.info(f"Модель сохранена в {config.MODEL_PATH}")
    print(f"✅ Обучение завершено! Модель сохранена в {config.MODEL_PATH}")
    print(f"📊 Параметры: {len(all_words)} слов, {len(tags)} намерений")

if __name__ == "__main__":
    train_model()
