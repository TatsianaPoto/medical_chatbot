"""Конфигурация бота для медицинского центра"""
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл
load_dotenv()

# Определяем корневую директорию проекта
BASE_DIR = Path(__file__).parent.absolute()


@dataclass
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

    # Пути к файлам (из .env или по умолчанию)
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "data.pth")
    INTENTS_FILE = BASE_DIR / os.getenv("INTENTS_FILE", "intents.json")

    # Параметры обучения (из .env)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
    HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "8"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1000"))
    MAX_LEN = int(os.getenv("MAX_LEN", "20"))
    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "5"))

    # GPU Configuration
    USE_CUDA = os.getenv("USE_CUDA", "True").lower() == "true"
    CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
    PIN_MEMORY = os.getenv("PIN_MEMORY", "True").lower() == "true"
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

    # Google Calendar
    GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    GOOGLE_CREDENTIALS_FILE = BASE_DIR / os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    GOOGLE_TOKEN_FILE = BASE_DIR / os.getenv("GOOGLE_TOKEN_FILE", "token.pickle")

    # Логирование
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / os.getenv("LOG_FILE", "logs/bot.log")

    # База данных записей - data/appointments.db
    DB_FILE = BASE_DIR / "data" / "appointments.db"

    # NLTK
    NLTK_DATA_PATH = BASE_DIR / os.getenv("NLTK_DATA_PATH", "nltk_data")

    # Часовой пояс
    TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow")

    @property
    def DEVICE(self):
        """Определение устройства для вычислений (GPU/CPU)"""
        if self.USE_CUDA and torch.cuda.is_available():
            return torch.device(f'cuda:{self.CUDA_DEVICE_ID}')
        else:
            return torch.device('cpu')


# Создаем экземпляр конфига
config = Config()

# Создаем необходимые директории
def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        config.LOG_FILE.parent,
        config.DB_FILE.parent,
        config.NLTK_DATA_PATH
    ]

    for directory in directories:
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Директория: {directory}")


# Вызываем создание директорий
setup_directories()

# Проверка наличия токена
def check_config():
    """Проверка конфигурации"""
    if not config.TELEGRAM_TOKEN:
        print("❌ ОШИБКА: TELEGRAM_TOKEN не найден в .env файле!")
        return False
    print(f"✅ Конфигурация загружена из .env")
    print(f"   - Устройство: {config.DEVICE}")
    print(f"   - Модель: {config.MODEL_PATH}")
    print(f"   - БД: {config.DB_FILE}")
    return True


# Экспорт для обратной совместимости
TELEGRAM_TOKEN = config.TELEGRAM_TOKEN
DEVICE = config.DEVICE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
INTENTS_FILE = config.INTENTS_FILE
MODEL_FILE = config.MODEL_PATH
LOG_FILE = config.LOG_FILE
CALENDAR_CREDENTIALS_FILE = config.GOOGLE_CREDENTIALS_FILE
CALENDAR_TOKEN_FILE = config.GOOGLE_TOKEN_FILE
CALENDAR_ID = config.GOOGLE_CALENDAR_ID

if __name__ == "__main__":
    check_config()