"""
Telegram бот для медицинского центра с использованием PyTorch NLP
Поддержка GPU, локальное хранение записей, логирование
"""
import random
import json
import torch
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    ConversationHandler, filters, ContextTypes
)

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from config import config
from logger import logger, conversation_logger
from calendar_manager import calendar_manager

# Состояния разговора для ConversationHandler
(
    STATE_IDLE,
    STATE_SELECT_SPECIALTY,
    STATE_SELECT_DATE,
    STATE_SELECT_TIME,
    STATE_GET_NAME,
    STATE_GET_PHONE,
    STATE_CONFIRM,
    STATE_CANCEL_SELECT
) = range(8)

class MedicalBot:
    """Медицинский чат-бот с поддержкой GPU"""
    
    def __init__(self):
        # Определяем устройство (GPU/CPU) из конфига
        self.device = config.DEVICE
        self.model = None
        self.all_words = None
        self.tags = None
        self.intents = None
        
        # Выводим информацию об устройстве
        try:
            if torch.cuda.is_available():
                logger.info(f"🚀 Бот использует GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("💻 Бот использует CPU")
        except Exception as e:
            logger.info(f"💻 Бот использует устройство: {self.device}")
        
        self.load_model()

    def load_model(self):
        """Загрузка обученной модели"""
        try:
            # Проверяем существование файла модели
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Файл модели {config.MODEL_PATH} не найден!")
            
            logger.info(f"Загрузка модели из {config.MODEL_PATH}")
            
            # Загружаем модель
            data = torch.load(config.MODEL_PATH, map_location=self.device)
            
            input_size = data["input_size"]
            hidden_size = data["hidden_size"]
            output_size = data["output_size"]
            self.all_words = data['all_words']
            self.tags = data['tags']
            model_state = data["model_state"]
            
            # Создаем и загружаем модель
            self.model = NeuralNet(input_size, hidden_size, output_size).to(self.device)
            self.model.load_state_dict(model_state)
            self.model.eval()
            
            # Загружаем intents
            with open(config.INTENTS_FILE, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            
            logger.info(f"✅ Модель успешно загружена")
            logger.info(f"   - Размер словаря: {len(self.all_words)} слов")
            logger.info(f"   - Количество намерений: {len(self.tags)}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            logger.error("   Пожалуйста, запустите train.py для обучения модели")
            raise
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при загрузке модели: {e}")
            conversation_logger.log_error(e, "load_model")
            raise

    def predict_intent(self, sentence: str) -> Tuple[str, float]:
        """Предсказание намерения пользователя"""
        try:
            # Токенизация и преобразование
            sentence_tokens = tokenize(sentence)
            X = bag_of_words(sentence_tokens, self.all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).float().to(self.device)
            
            # Предсказание
            with torch.no_grad():
                output = self.model(X)
                _, predicted = torch.max(output, dim=1)
                
                # Получаем вероятность
                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()].item()
                
            tag = self.tags[predicted.item()]
            return tag, prob
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании намерения: {e}")
            return "unknown", 0.0

    def get_response(self, tag: str, confidence: float) -> str:
        """Получение ответа по тегу намерения"""
        # Если уверенность низкая, используем fallback
        if confidence < 0.75:
            tag = "no_match"
            logger.debug(f"Низкая уверенность ({confidence:.2%}), используем fallback")
        
        # Ищем ответ для тега
        for intent in self.intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return response
        
        # Ответ по умолчанию
        return "Извините, я не совсем понял. Можете переформулировать вопрос?"

# Инициализация бота
try:
    bot = MedicalBot()
    logger.info("✅ Бот успешно инициализирован")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации бота: {e}")
    raise

# Хранилище данных пользователей (только для активной записи)
user_data: Dict[int, Dict[str, Any]] = {}

# ============ КЛАВИАТУРЫ ============
specialty_keyboard = ReplyKeyboardMarkup(
    [
        ['Терапевт', 'Кардиолог', 'Невролог'],
        ['Хирург', 'Офтальмолог', 'ЛОР'],
        ['Гинеколог', 'Эндокринолог', 'Педиатр'],
        ['❌ Отмена']
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
    input_field_placeholder="Выберите специальность врача"
)

time_keyboard = ReplyKeyboardMarkup(
    [
        ['09:00', '10:00', '11:00', '12:00'],
        ['13:00', '14:00', '15:00', '16:00'],
        ['17:00', '18:00', '19:00'],
        ['Другое время', '❌ Отмена']
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
    input_field_placeholder="Выберите удобное время"
)

confirm_keyboard = ReplyKeyboardMarkup(
    [
        ['✅ Да, подтверждаю'],
        ['❌ Нет, отменить'],
        ['📝 Изменить данные']
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
    input_field_placeholder="Подтвердите или измените данные"
)

# ============ ОБРАБОТЧИКИ КОМАНД ============

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка команды /start"""
    user = update.effective_user
    
    conversation_logger.log_command(user.id, user.username or "Unknown", "start")
    logger.info(f"Пользователь {user.id} (@{user.username}) запустил бота")
    
    welcome_text = (
        f"👋 Здравствуйте, {user.first_name}!\n\n"
        "🤖 Я медицинский ассистент клиники «Здоровье+».\n\n"
        "Я могу:\n"
        "✅ Записать вас к врачу\n"
        "✅ Рассказать о работе клиники\n"
        "✅ Показать ваши записи\n"
        "✅ Отменить или перенести прием\n\n"
        "🏥 <b>О клинике:</b>\n"
        "📍 Адрес: ул. Медицинская, д. 15\n"
        "⏰ Часы работы: Пн-Пт 8:00-20:00, Сб 9:00-15:00\n"
        "📞 Телефон: +7 (495) 123-45-67\n\n"
        "💡 <b>Как записаться:</b>\n"
        "Напишите «Записаться к врачу» или нажмите на кнопку меню"
    )
    
    await update.message.reply_text(welcome_text, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, welcome_text)
    
    return STATE_IDLE

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /help"""
    user = update.effective_user
    
    conversation_logger.log_command(user.id, user.username or "Unknown", "help")
    
    help_text = (
        "🔹 <b>Доступные команды:</b>\n\n"
        "/start - Начать диалог\n"
        "/help - Показать эту справку\n"
        "/myappointments - Мои записи\n"
        "/cancel - Отменить запись\n"
        "/contacts - Контакты клиники\n\n"
        "<b>💬 Что я умею:</b>\n"
        "• Записывать к врачу\n"
        "• Рассказывать о врачах\n"
        "• Отвечать на вопросы о клинике\n"
        "• Помогать отменить запись\n\n"
        "<b>🩺 Доступные специалисты:</b>\n"
        "Терапевт, Кардиолог, Невролог, Хирург,\n"
        "Офтальмолог, ЛОР, Гинеколог, Эндокринолог,\n"
        "Педиатр\n\n"
        "<b>📝 Примеры запросов:</b>\n"
        "• «Записаться к терапевту»\n"
        "• «Хочу записаться на прием»\n"
        "• «Когда вы работаете?»\n"
        "• «Какой у вас адрес?»"
    )
    
    await update.message.reply_text(help_text, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, help_text[:100] + "...")

async def contacts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /contacts"""
    user = update.effective_user
    
    conversation_logger.log_command(user.id, user.username or "Unknown", "contacts")
    
    contacts_text = (
        "📞 <b>Контакты медицинского центра</b>\n\n"
        "📍 <b>Адрес:</b> г. Минск, ул. Медицинская, д. 15\n"
        "🚇 <b>Метро:</b> «Академическая» (5 мин пешком)\n\n"
        "📅 <b>Режим работы:</b>\n"
        "Понедельник-пятница: 8:00 - 20:00\n"
        "Суббота: 9:00 - 15:00\n"
        "Воскресенье: выходной\n\n"
        "📞 <b>Телефоны:</b>\n"
        "Регистратура: +7 (495) 123-45-67\n"
        "Справочная: +7 (495) 123-45-68\n"
        "Экстренная помощь: 103\n\n"
        "✉️ <b>Email:</b> info@medclinic.ru\n\n"
        "🌐 <b>Сайт:</b> www.medclinic.ru"
    )
    
    await update.message.reply_text(contacts_text, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, contacts_text[:100] + "...")

async def my_appointments(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать записи пользователя"""
    user = update.effective_user
    
    conversation_logger.log_command(user.id, user.username or "Unknown", "myappointments")
    logger.info(f"Пользователь {user.id} запросил список записей")
    
    # Получаем записи из менеджера календаря
    appointments = calendar_manager.get_user_appointments(user.id)
    
    if not appointments:
        response = (
            "📋 У вас пока нет активных записей.\n\n"
            "Хотите записаться к врачу? Напишите «Записаться к врачу»"
        )
        await update.message.reply_text(response)
        conversation_logger.log_bot_response(user.id, response)
        return
    
    text = "📋 <b>Ваши активные записи:</b>\n\n"
    for i, app in enumerate(appointments, 1):
        text += (
            f"{i}. <b>{app['doctor']}</b>\n"
            f"   👤 {app['patient_name']}\n"
            f"   📅 {app['date']}\n"
            f"   🕐 {app['time']}\n"
            f"   📞 {app['phone']}\n\n"
        )
    
    text += "\n❓ Чтобы отменить запись, используйте команду /cancel"
    
    await update.message.reply_text(text, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, text[:100] + "...")

# ============ ОТМЕНА ЗАПИСИ С ВЫБОРОМ ============

async def cancel_appointment_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отмена записи - показывает список записей для выбора"""
    user = update.effective_user
    
    conversation_logger.log_command(user.id, user.username or "Unknown", "cancel")
    logger.info(f"Пользователь {user.id} хочет отменить запись")
    
    # Получаем активные записи пользователя
    appointments = calendar_manager.get_user_appointments(user.id)
    
    if not appointments:
        response = (
            "❌ У вас нет активных записей для отмены.\n\n"
            "Чтобы записаться, напишите «Записаться к врачу»"
        )
        await update.message.reply_text(response)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_IDLE
    
    # Создаем клавиатуру со списком записей
    cancel_buttons = []
    for i, app in enumerate(appointments, 1):
        button_text = f"{i}. {app['doctor']} - {app['date']} {app['time']}"
        cancel_buttons.append([button_text])
    
    cancel_buttons.append(['❌ Отмена'])
    
    cancel_keyboard = ReplyKeyboardMarkup(
        cancel_buttons,
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Выберите запись для отмены"
    )
    
    # Сохраняем список записей в context.user_data
    context.user_data['cancel_appointments'] = appointments
    
    response = "📋 <b>Ваши активные записи:</b>\n\n"
    for i, app in enumerate(appointments, 1):
        response += (
            f"{i}. 🩺 <b>{app['doctor']}</b>\n"
            f"   📅 {app['date']} {app['time']}\n"
            f"   👤 {app['patient_name']}\n\n"
        )
    
    response += "Выберите номер записи, которую хотите отменить:"
    
    await update.message.reply_text(
        response,
        reply_markup=cancel_keyboard,
        parse_mode='HTML'
    )
    
    conversation_logger.log_bot_response(user.id, "Показан список записей для отмены")
    
    return STATE_CANCEL_SELECT

async def cancel_select(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора записи для отмены"""
    user = update.effective_user
    text = update.message.text.strip()
    
    logger.info(f"cancel_select: user={user.id}, text={text}")
    
    if text == '❌ Отмена':
        response = "❌ Отмена операции. Ваши записи сохранены."
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        conversation_logger.log_bot_response(user.id, response)
        context.user_data.pop('cancel_appointments', None)
        return STATE_IDLE
    
    # Получаем сохраненный список записей
    appointments = context.user_data.get('cancel_appointments', [])
    
    if not appointments:
        response = "❌ Произошла ошибка. Пожалуйста, попробуйте снова."
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        return STATE_IDLE
    
    # Пытаемся извлечь номер записи из текста
    match = re.match(r'^(\d+)\.', text)
    if match:
        selected_num = int(match.group(1))
    else:
        try:
            selected_num = int(text.split('.')[0])
        except ValueError:
            response = "❌ Пожалуйста, выберите номер записи из списка, используя кнопки."
            await update.message.reply_text(response)
            return STATE_CANCEL_SELECT
    
    # Проверяем корректность номера
    if selected_num < 1 or selected_num > len(appointments):
        response = f"❌ Неверный номер. Пожалуйста, выберите число от 1 до {len(appointments)}."
        await update.message.reply_text(response)
        return STATE_CANCEL_SELECT
    
    # Получаем выбранную запись
    selected_appointment = appointments[selected_num - 1]
    appointment_id = selected_appointment.get('id')
    
    logger.info(f"Пользователь {user.id} выбрал запись #{appointment_id} для отмены")
    
    # Отменяем выбранную запись
    success = calendar_manager.cancel_appointment(user.id, appointment_id)
    
    # Добавляем отладочный вывод
    logger.info(f"Результат отмены: {success}")
    
    if success:
        # Проверяем, что запись действительно отменена
        check_appointment = calendar_manager.get_appointment_by_id(appointment_id)
        if check_appointment:
            logger.info(f"Статус записи после отмены: {check_appointment.get('status')}")
        
        response = (
            f"✅ <b>Запись успешно отменена!</b>\n\n"
            f"🩺 Врач: {selected_appointment['doctor']}\n"
            f"📅 Дата: {selected_appointment['date']}\n"
            f"🕐 Время: {selected_appointment['time']}\n\n"
            f"Если хотите записаться снова, напишите «Записаться к врачу»"
        )
        conversation_logger.log_cancellation(user.id, appointment_id)
        logger.info(f"✅ Пользователь {user.id} отменил запись #{appointment_id}")
    else:
        response = "❌ Не удалось отменить запись. Пожалуйста, попробуйте позже."
        logger.error(f"❌ Не удалось отменить запись #{appointment_id} для пользователя {user.id}")
    
    await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, response[:200])
    
    # Очищаем временные данные
    context.user_data.pop('cancel_appointments', None)
    
    return STATE_IDLE

# ============ ОСНОВНОЙ ОБРАБОТЧИК СООБЩЕНИЙ ============

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Основной обработчик сообщений (не в режиме разговора)"""
    user = update.effective_user
    message_text = update.message.text
    
    # Предсказываем намерение
    tag, confidence = bot.predict_intent(message_text)
    
    # Логируем сообщение
    conversation_logger.log_message(
        user.id, 
        user.username or "Unknown", 
        message_text, 
        tag, 
        confidence
    )
    
    # Проверяем, хочет ли пользователь записаться
    if tag in ["book_appointment", "select_specialty"] or \
       any(word in message_text.lower() for word in ["записаться", "запись", "к врачу", "на прием"]):
        logger.info(f"Пользователь {user.id} инициировал запись")
        return await start_booking(update, context)
    
    # Проверяем, хочет ли пользователь отменить запись
    if tag == "cancel_appointment" or any(word in message_text.lower() for word in ["отменить", "отмена"]):
        return await cancel_appointment_command(update, context)
    
    # Получаем обычный ответ
    response = bot.get_response(tag, confidence)
    
    await update.message.reply_text(response)
    conversation_logger.log_bot_response(user.id, response, tag)
    
    return STATE_IDLE

# ============ СЦЕНАРИЙ ЗАПИСИ ============

async def start_booking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало процесса записи"""
    user = update.effective_user
    
    logger.info(f"Пользователь {user.id} начал процесс записи")
    
    # Очищаем старые данные пользователя, если есть
    if user.id in user_data:
        del user_data[user.id]
    
    # Инициализируем новые данные
    user_data[user.id] = {
        'specialty': None,
        'date': None,
        'date_str': None,
        'time': None,
        'patient_name': None,
        'phone': None,
        'user_id': user.id,
        'username': user.username
    }
    
    response = (
        "🏥 <b>Запись к врачу</b>\n\n"
        "Пожалуйста, выберите специальность врача, нажав на кнопку ниже:\n\n"
        "Доступные врачи:\n"
        "• Терапевт - общие заболевания\n"
        "• Кардиолог - сердце и сосуды\n"
        "• Невролог - нервная система\n"
        "• Хирург - операции\n"
        "• Офтальмолог - глаза\n"
        "• ЛОР - ухо, горло, нос\n"
        "• Гинеколог - женское здоровье\n"
        "• Эндокринолог - гормоны\n"
        "• Педиатр - детский врач"
    )
    
    await update.message.reply_text(
        response,
        reply_markup=specialty_keyboard,
        parse_mode='HTML'
    )
    
    conversation_logger.log_bot_response(user.id, "Запрос на выбор специальности")
    
    return STATE_SELECT_SPECIALTY

async def select_specialty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор специальности врача"""
    user = update.effective_user
    text = update.message.text.strip()
    
    logger.info(f"select_specialty: user={user.id}, text={text}")
    
    if text == '❌ Отмена':
        response = "Запись отменена. Если передумаете - обращайтесь!"
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        conversation_logger.log_bot_response(user.id, response)
        if user.id in user_data:
            del user_data[user.id]
        return STATE_IDLE
    
    available_specialties = ['Терапевт', 'Кардиолог', 'Невролог', 'Хирург', 
                            'Офтальмолог', 'ЛОР', 'Гинеколог', 'Эндокринолог', 'Педиатр']
    
    if text not in available_specialties:
        response = f"❌ Пожалуйста, выберите специальность из предложенных кнопок:\n\n{', '.join(available_specialties)}"
        await update.message.reply_text(response, reply_markup=specialty_keyboard)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_SPECIALTY
    
    user_data[user.id]['specialty'] = text
    logger.info(f"Пользователь {user.id} выбрал специальность: {text}")
    
    # Генерируем доступные даты
    dates = []
    today = datetime.now()
    
    for i in range(14):
        date = today + timedelta(days=i)
        if date.weekday() != 6:
            date_str = date.strftime("%d.%m.%Y")
            day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
            day_name = day_names[date.weekday()]
            dates.append(f"{date_str} ({day_name})")
    
    date_buttons = []
    for date in dates[:10]:
        date_buttons.append([date])
    date_buttons.append(['❌ Отмена'])
    
    date_keyboard = ReplyKeyboardMarkup(date_buttons, resize_keyboard=True, one_time_keyboard=False)
    
    response = f"✅ Вы выбрали: <b>{text}</b>\n\n📅 Теперь выберите удобную дату приема:"
    
    await update.message.reply_text(
        response,
        reply_markup=date_keyboard,
        parse_mode='HTML'
    )
    
    conversation_logger.log_bot_response(user.id, response)
    
    return STATE_SELECT_DATE

async def select_date(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор даты приема"""
    user = update.effective_user
    text = update.message.text
    
    logger.info(f"select_date: user={user.id}, text={text}")
    
    if text == '❌ Отмена':
        response = "Запись отменена."
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        conversation_logger.log_bot_response(user.id, response)
        if user.id in user_data:
            del user_data[user.id]
        return STATE_IDLE
    
    date_str = text.split(' (')[0]
    
    try:
        date_obj = datetime.strptime(date_str, "%d.%m.%Y")
        
        if date_obj.date() < datetime.now().date():
            response = "❌ Нельзя выбрать дату в прошлом.\n\nПожалуйста, выберите будущую дату."
            await update.message.reply_text(response)
            conversation_logger.log_bot_response(user.id, response)
            return STATE_SELECT_DATE
        
        user_data[user.id]['date'] = date_obj.strftime("%Y-%m-%d")
        user_data[user.id]['date_str'] = date_str
        
        logger.info(f"Пользователь {user.id} выбрал дату: {date_str}")
        
        response = f"📅 Дата: <b>{date_str}</b>\n\n🕐 Теперь выберите удобное время:"
        
        await update.message.reply_text(
            response,
            reply_markup=time_keyboard,
            parse_mode='HTML'
        )
        
        conversation_logger.log_bot_response(user.id, response)
        
        return STATE_SELECT_TIME
        
    except ValueError as e:
        logger.error(f"Ошибка парсинга даты: {e}")
        response = "❌ Пожалуйста, выберите дату из предложенных вариантов."
        await update.message.reply_text(response)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_DATE

async def select_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Выбор времени приема"""
    user = update.effective_user
    text = update.message.text
    
    logger.info(f"select_time: user={user.id}, text={text}")
    
    if text == '❌ Отмена':
        response = "Запись отменена."
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        conversation_logger.log_bot_response(user.id, response)
        if user.id in user_data:
            del user_data[user.id]
        return STATE_IDLE
    
    if text == 'Другое время':
        response = "⌨️ Пожалуйста, напишите удобное время в формате <b>ЧЧ:ММ</b>\nНапример: 14:30 или 09:15\n\n⏰ Часы работы: 9:00 - 20:00"
        await update.message.reply_text(response, parse_mode='HTML')
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_TIME
    
    time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$')
    if not time_pattern.match(text):
        response = "❌ Неверный формат времени.\n\nПожалуйста, выберите время из предложенных вариантов или нажмите «Другое время»"
        await update.message.reply_text(response, reply_markup=time_keyboard)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_TIME
    
    data = user_data[user.id]
    available_times = calendar_manager.get_available_times(data['specialty'], data['date_str'])
    
    if text not in available_times:
        times_display = ', '.join(available_times[:8]) if available_times else "нет свободного времени"
        response = (f"❌ Время {text} уже занято.\n\n"
                   f"📅 Доступное время на {data['date_str']}:\n"
                   f"{times_display}\n\n"
                   "Пожалуйста, выберите другое время:")
        await update.message.reply_text(response, reply_markup=time_keyboard)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_TIME
    
    user_data[user.id]['time'] = text
    
    logger.info(f"Пользователь {user.id} выбрал время: {text}")
    
    response = (f"🕐 Время: <b>{text}</b>\n\n"
                f"👤 Теперь напишите ваше <b>полное имя и фамилию</b>\n"
                f"Например: Иван Петров\n\n"
                f"💡 Это нужно для оформления записи")
    
    await update.message.reply_text(response, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, response)
    
    return STATE_GET_NAME

async def get_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение имени пациента"""
    user = update.effective_user
    name = update.message.text.strip()
    
    logger.info(f"get_name: user={user.id}, name={name}")
    
    if len(name) < 3:
        response = "❌ Пожалуйста, укажите полное имя и фамилию (минимум 3 символа)\n\nПример: Иван Петров"
        await update.message.reply_text(response)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_GET_NAME
    
    user_data[user.id]['patient_name'] = name
    
    logger.info(f"Пользователь {user.id} указал имя: {name}")
    
    response = (f"👤 Имя: <b>{name}</b>\n\n"
                f"📞 Теперь напишите ваш <b>номер телефона</b>\n"
                f"Например: +7 123 456-78-90 или 89123456789\n\n"
                f"💡 Нужен для связи и подтверждения записи")
    
    await update.message.reply_text(response, parse_mode='HTML')
    conversation_logger.log_bot_response(user.id, response)
    
    return STATE_GET_PHONE

async def get_phone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение телефона пациента"""
    user = update.effective_user
    phone = update.message.text.strip()
    
    phone_clean = re.sub(r'[^\d+]', '', phone)
    
    logger.info(f"get_phone: user={user.id}, phone_clean={phone_clean}")
    
    if len(phone_clean) < 10:
        response = "❌ Пожалуйста, укажите корректный номер телефона\n\nПример: +7 123 456-78-90\n\nДолжно быть минимум 10 цифр"
        await update.message.reply_text(response)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_GET_PHONE
    
    user_data[user.id]['phone'] = phone_clean
    
    logger.info(f"Пользователь {user.id} указал телефон: {phone_clean}")
    
    data = user_data[user.id]
    
    confirm_text = (
        "📋 <b>Проверьте данные записи:</b>\n\n"
        f"🩺 Врач: {data['specialty']}\n"
        f"👤 Пациент: {data['patient_name']}\n"
        f"📞 Телефон: {data['phone']}\n"
        f"📅 Дата: {data['date_str']}\n"
        f"🕐 Время: {data['time']}\n\n"
        "✅ Все верно?\n\n"
        "Если нужно изменить данные, нажмите «Изменить данные»"
    )
    
    await update.message.reply_text(
        confirm_text,
        reply_markup=confirm_keyboard,
        parse_mode='HTML'
    )
    
    conversation_logger.log_bot_response(user.id, "Показана форма подтверждения")
    
    return STATE_CONFIRM

async def confirm_booking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Подтверждение или изменение записи"""
    user = update.effective_user
    text = update.message.text
    
    logger.info(f"confirm_booking: user={user.id}, text={text}")
    
    if text == '❌ Нет, отменить':
        response = "❌ Запись отменена.\n\nЕсли передумаете - просто напишите «Записаться к врачу»!"
        await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
        conversation_logger.log_bot_response(user.id, response)
        if user.id in user_data:
            del user_data[user.id]
        return STATE_IDLE
    
    if text == '📝 Изменить данные':
        if user.id in user_data:
            del user_data[user.id]
        response = "🔄 Давайте начнем заново.\n\nВыберите специальность врача:"
        await update.message.reply_text(response, reply_markup=specialty_keyboard)
        conversation_logger.log_bot_response(user.id, response)
        return STATE_SELECT_SPECIALTY
    
    if text == '✅ Да, подтверждаю':
        data = user_data.get(user.id)
        
        if not data or not data.get('specialty'):
            response = "❌ Произошла ошибка. Пожалуйста, начните запись заново."
            await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove())
            conversation_logger.log_bot_response(user.id, response)
            if user.id in user_data:
                del user_data[user.id]
            return STATE_IDLE
        
        # Проверяем еще раз доступность времени
        available_times = calendar_manager.get_available_times(data['specialty'], data['date_str'])
        if data['time'] not in available_times:
            response = (f"❌ <b>К сожалению, время {data['time']} уже занято</b>\n\n"
                       f"Пожалуйста, начните запись заново и выберите другое время.")
            await update.message.reply_text(response, reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')
            conversation_logger.log_bot_response(user.id, response)
            if user.id in user_data:
                del user_data[user.id]
            return STATE_IDLE
        
        success = calendar_manager.add_appointment(
            patient_name=data['patient_name'],
            doctor=data['specialty'],
            date_str=data['date_str'],
            time_str=data['time'],
            phone=data['phone'],
            user_id=data['user_id'],
            username=data.get('username')
        )
        
        if success:
            response = (
                "✅ <b>Запись успешно подтверждена!</b>\n\n"
                f"🩺 Врач: {data['specialty']}\n"
                f"👤 Пациент: {data['patient_name']}\n"
                f"📅 Дата: {data['date_str']}\n"
                f"🕐 Время: {data['time']}\n\n"
                "📍 <b>Как нас найти:</b>\n"
                "г. Минск, ул. Медицинская, д. 15\n\n"
                "📋 <b>Что взять с собой:</b>\n"
                "• Паспорт\n"
                "• Полис ОМС\n\n"
                "⏰ Приходите за 10-15 минут до приема\n\n"
                "❓ Отменить запись можно командой /cancel\n\n"
                "🙂 Спасибо, что выбрали нашу клинику!"
            )
            logger.info(f"✅ Создана запись для пользователя {user.id}")
            conversation_logger.log_appointment(user.id, data)
        else:
            response = (
                "❌ <b>Извините, не удалось создать запись</b>\n\n"
                "Возможно, это время уже занято.\n\n"
                "Пожалуйста, попробуйте снова или позвоните нам:\n"
                "📞 +7 (495) 123-45-67"
            )
            logger.error(f"❌ Ошибка создания записи для пользователя {user.id}")
        
        await update.message.reply_text(
            response,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode='HTML'
        )
        
        conversation_logger.log_bot_response(user.id, response[:200] + "...")
        
        if user.id in user_data:
            del user_data[user.id]
        
        return STATE_IDLE
    
    response = "Пожалуйста, используйте кнопки для подтверждения или изменения."
    await update.message.reply_text(response, reply_markup=confirm_keyboard)
    return STATE_CONFIRM

# ============ ОБРАБОТКА ОШИБОК ============

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок"""
    error = context.error
    user_id = update.effective_user.id if update and update.effective_user else None
    
    conversation_logger.log_error(error, "error_handler", user_id)
    logger.error(f"Update {update} caused error {error}")
    
    if update and update.effective_message:
        response = (
            "❌ Произошла техническая ошибка.\n\n"
            "Наши специалисты уже уведомлены. Пожалуйста, попробуйте позже "
            "или свяжитесь с нами по телефону: +7 (495) 123-45-67"
        )
        await update.effective_message.reply_text(response)

# ============ ЗАПУСК БОТА ============

def main() -> None:
    """Запуск Telegram бота"""
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК MEDICAL CHAT BOT")
    logger.info("=" * 60)
    
    # Создаем приложение
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    
    # ConversationHandler для сценария записи
    booking_conversation = ConversationHandler(
        entry_points=[
            MessageHandler(filters.Regex(r'(?i)(записаться|запись|прием|к врачу|консультация)'), start_booking),
            CommandHandler("book", start_booking)
        ],
        states={
            STATE_SELECT_SPECIALTY: [
                MessageHandler(filters.Regex(r'^(Терапевт|Кардиолог|Невролог|Хирург|Офтальмолог|ЛОР|Гинеколог|Эндокринолог|Педиатр|❌ Отмена)$'), select_specialty)
            ],
            STATE_SELECT_DATE: [
                MessageHandler(filters.Regex(r'^\d{2}\.\d{2}\.\d{4}|❌ Отмена$'), select_date)
            ],
            STATE_SELECT_TIME: [
                MessageHandler(filters.Regex(r'^(\d{2}:\d{2}|Другое время|❌ Отмена)$'), select_time)
            ],
            STATE_GET_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, get_name)
            ],
            STATE_GET_PHONE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, get_phone)
            ],
            STATE_CONFIRM: [
                MessageHandler(filters.Regex(r'^(✅ Да, подтверждаю|❌ Нет, отменить|📝 Изменить данные)$'), confirm_booking)
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel_appointment_command),
            CommandHandler("start", start)
        ],
        name="booking_conversation",
        persistent=False,
        allow_reentry=True
    )
    
    # ConversationHandler для отмены записи
    cancel_conversation = ConversationHandler(
        entry_points=[
            CommandHandler("cancel", cancel_appointment_command),
            MessageHandler(filters.Regex(r'(?i)(отменить|отмена)'), cancel_appointment_command)
        ],
        states={
            STATE_CANCEL_SELECT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, cancel_select)
            ],
        },
        fallbacks=[
            CommandHandler("start", start),
            CommandHandler("cancel", cancel_appointment_command)
        ],
        name="cancel_conversation",
        persistent=False,
        allow_reentry=True
    )
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("contacts", contacts_command))
    application.add_handler(CommandHandler("myappointments", my_appointments))
    application.add_handler(booking_conversation)
    application.add_handler(cancel_conversation)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    logger.info("✅ Бот успешно запущен и готов к работе!")
    logger.info(f"📡 Режим: polling")
    logger.info(f"📁 Файл БД: {config.DB_FILE}")
    logger.info(f"📁 Модель: {config.MODEL_PATH}")
    
    try:
        if torch.cuda.is_available():
            logger.info(f"🖥️ Устройство: GPU ({torch.cuda.get_device_name(0)})")
        else:
            logger.info("🖥️ Устройство: CPU")
    except:
        logger.info("🖥️ Устройство: CPU")
    
    logger.info("=" * 60)
    print("\n✅ Бот запущен! Найдите его в Telegram\n")
    print("📋 Доступные команды:")
    print("   /start - Начать работу")
    print("   /help - Помощь")
    print("   /myappointments - Мои записи")
    print("   /cancel - Отменить запись")
    print("   /contacts - Контакты\n")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()