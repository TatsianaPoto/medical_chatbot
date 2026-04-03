"""Логирование работы бота"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from config import config

def setup_logger(name: str = "medical_bot") -> logging.Logger:
    """Настройка логгера с записью в файл и консоль"""
    
    # Создаем директорию для логов, если её нет
    log_file = Path(config.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Очищаем существующие обработчики, чтобы не дублировать
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Форматирование с подробной информацией
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для файла с ротацией (максимум 10 MB)
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.LOG_FILE, 
            encoding='utf-8',
            maxBytes=10_485_760,  # 10 MB
            backupCount=5
        )
    except:
        # Fallback на обычный FileHandler
        file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Обработчик для консоли (только INFO и выше)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Добавляем обработчики
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class ConversationLogger:
    """Логирование диалогов с пользователями"""
    
    def __init__(self):
        self.logger = setup_logger("conversations")
        self.conversations: Dict[int, list] = {}
        self.max_history = 100  # Максимальное количество сообщений на пользователя
    
    def log_message(
        self, 
        user_id: int, 
        username: str, 
        message: str, 
        intent: Optional[str] = None, 
        confidence: Optional[float] = None
    ):
        """
        Логирование сообщения пользователя
        
        Args:
            user_id: ID пользователя в Telegram
            username: Имя пользователя
            message: Текст сообщения
            intent: Определенное намерение (опционально)
            confidence: Уверенность модели (опционально)
        """
        # Формируем сообщение для лога
        log_message = f"User [{user_id}] @{username}: {message}"
        if intent:
            log_message += f" | Intent: {intent}"
        if confidence:
            log_message += f" | Confidence: {confidence:.2%}"
        
        self.logger.info(log_message)
        
        # Сохраняем контекст разговора
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        # Ограничиваем историю
        if len(self.conversations[user_id]) >= self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
        
        self.conversations[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "message": message,
            "intent": intent,
            "confidence": confidence
        })
    
    def log_bot_response(self, user_id: int, response: str, tag: Optional[str] = None):
        """
        Логирование ответа бота
        
        Args:
            user_id: ID пользователя
            response: Текст ответа
            tag: Тег намерения (опционально)
        """
        log_message = f"Bot -> User [{user_id}]: {response[:200]}"  # Ограничиваем длину
        if tag:
            log_message += f" | Tag: {tag}"
        
        self.logger.info(log_message)
        
        # Сохраняем контекст разговора
        if user_id in self.conversations:
            self.conversations[user_id].append({
                "timestamp": datetime.now().isoformat(),
                "role": "bot",
                "message": response,
                "tag": tag
            })
    
    def log_appointment(self, user_id: int, details: Dict[str, Any]):
        """
        Логирование записи на прием
        
        Args:
            user_id: ID пользователя
            details: Детали записи (врач, дата, время и т.д.)
        """
        self.logger.info(
            f"📅 APPOINTMENT | User [{user_id}] | "
            f"Doctor: {details.get('specialty', 'N/A')} | "
            f"Date: {details.get('date_str', 'N/A')} | "
            f"Time: {details.get('time', 'N/A')} | "
            f"Patient: {details.get('patient_name', 'N/A')}"
        )
    
    def log_cancellation(self, user_id: int, appointment_id: Optional[int] = None):
        """
        Логирование отмены записи
        
        Args:
            user_id: ID пользователя
            appointment_id: ID записи (опционально)
        """
        message = f"❌ CANCELLATION | User [{user_id}]"
        if appointment_id:
            message += f" | Appointment ID: {appointment_id}"
        self.logger.info(message)
    
    def log_error(self, error: Exception, context: str = "", user_id: Optional[int] = None):
        """
        Логирование ошибок
        
        Args:
            error: Объект исключения
            context: Контекст ошибки
            user_id: ID пользователя (если есть)
        """
        user_str = f"User [{user_id}] " if user_id else ""
        self.logger.error(
            f"❌ ERROR {user_str}| Context: {context} | Error: {str(error)}", 
            exc_info=True
        )
    
    def log_command(self, user_id: int, username: str, command: str):
        """
        Логирование выполнения команды
        
        Args:
            user_id: ID пользователя
            username: Имя пользователя
            command: Выполненная команда
        """
        self.logger.info(f"Command [{user_id}] @{username}: /{command}")
    
    def get_conversation_history(self, user_id: int, limit: int = 10) -> list:
        """
        Получение истории разговора пользователя
        
        Args:
            user_id: ID пользователя
            limit: Количество последних сообщений
            
        Returns:
            list: История сообщений
        """
        if user_id not in self.conversations:
            return []
        return self.conversations[user_id][-limit:]
    
    def clear_conversation(self, user_id: int):
        """
        Очистка истории разговора пользователя
        
        Args:
            user_id: ID пользователя
        """
        if user_id in self.conversations:
            del self.conversations[user_id]
            self.logger.debug(f"Cleared conversation history for user {user_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по диалогам
        
        Returns:
            dict: Статистика
        """
        total_messages = 0
        for history in self.conversations.values():
            total_messages += len(history)
        
        return {
            "total_users": len(self.conversations),
            "total_messages": total_messages,
            "active_users": len([u for u, h in self.conversations.items() if len(h) > 0])
        }

# Глобальные экземпляры
logger = setup_logger()
conversation_logger = ConversationLogger()

# Функция для логирования входящих запросов (декоратор)
def log_async_function(func):
    """Декоратор для логирования асинхронных функций"""
    async def wrapper(*args, **kwargs):
        try:
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = await func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}", exc_info=True)
            raise
    return wrapper

# Функция для логирования синхронных функций
def log_function(func):
    """Декоратор для логирования синхронных функций"""
    def wrapper(*args, **kwargs):
        try:
            logger.debug(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}", exc_info=True)
            raise
    return wrapper

# Инициализация при запуске
if __name__ == "__main__":
    # Тестовый вывод
    print("Testing logger...")
    logger.info("Logger initialized successfully")
    logger.warning("This is a test warning")
    logger.error("This is a test error")
    
    conversation_logger.log_message(123456, "test_user", "Hello, bot!", "greeting", 0.95)
    conversation_logger.log_bot_response(123456, "Hello! How can I help you?", "greeting")
    conversation_logger.log_appointment(123456, {
        "specialty": "Терапевт",
        "date_str": "05.04.2024",
        "time": "10:00",
        "patient_name": "Иван Петров"
    })
    
    print(f"Statistics: {conversation_logger.get_statistics()}")
    print("Logger test completed!")