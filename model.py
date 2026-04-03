"""Нейронная сеть для классификации намерений"""
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    Простая нейронная сеть с двумя скрытыми слоями
    для классификации намерений пользователя
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Args:
            input_size: размер входного слоя (количество слов в словаре)
            hidden_size: размер скрытого слоя
            num_classes: количество классов (намерений)
        """
        super(NeuralNet, self).__init__()

        # Первый скрытый слой
        self.l1 = nn.Linear(input_size, hidden_size)
        # Второй скрытый слой
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Выходной слой
        self.l3 = nn.Linear(hidden_size, num_classes)

        # Функция активации ReLU
        self.relu = nn.ReLU()
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Прямое распространение"""
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.l3(out)
        # Не применяем softmax здесь, так как используем CrossEntropyLoss
        return out
