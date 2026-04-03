"""
Менеджер календаря для медицинского бота
Сохраняет записи в SQLite базу данных data/appointments.db
"""
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from config import config


class CalendarManager:
    """Менеджер для работы с записями (SQLite хранение)"""

    def __init__(self):
        """Инициализация менеджера записей"""
        self.db_file = config.DB_FILE
        self._ensure_database()
        print(f"✅ Инициализирован менеджер записей (SQLite): {self.db_file}")

    def _is_valid_sqlite_db(self) -> bool:
        """Проверяет, является ли файл корректной SQLite базой данных"""
        if not self.db_file.exists():
            return False
        
        try:
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            cursor.fetchone()
            conn.close()
            return True
        except (sqlite3.DatabaseError, sqlite3.Error):
            return False

    def _ensure_database(self):
        """Создает таблицу если она не существует или файл поврежден"""
        # Создаем директорию если нужно
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Если файл существует, но не является SQLite БД - удаляем его
        if self.db_file.exists() and not self._is_valid_sqlite_db():
            print(f"⚠️ Файл {self.db_file} поврежден или не является SQLite БД. Удаляем...")
            os.remove(self.db_file)
            print(f"✅ Старый файл удален")
        
        # Создаем новую БД с таблицей
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_name TEXT NOT NULL,
                    doctor TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    user_id INTEGER,
                    username TEXT,
                    created_at TEXT,
                    cancelled_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            conn.commit()
            print(f"✅ Таблица appointments создана/проверена в {self.db_file}")

    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для подключения к БД"""
        conn = sqlite3.connect(str(self.db_file))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_appointment(self, patient_name: str, doctor: str, date_str: str,
                        time_str: str, phone: str, user_id: Optional[int] = None,
                        username: Optional[str] = None) -> bool:
        """Добавление новой записи"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Проверяем, нет ли уже записи на это время
                cursor.execute('''
                    SELECT id FROM appointments 
                    WHERE doctor = ? AND date = ? AND time = ? AND status = 'active'
                ''', (doctor, date_str, time_str))

                if cursor.fetchone():
                    print(f"⚠️ Время {date_str} {time_str} для врача {doctor} уже занято")
                    return False

                # Добавляем новую запись
                cursor.execute('''
                    INSERT INTO appointments 
                    (patient_name, doctor, date, time, phone, user_id, username, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    patient_name, doctor, date_str, time_str, phone,
                    user_id, username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'active'
                ))

                conn.commit()
                appointment_id = cursor.lastrowid

                print(f"✅ Создана запись #{appointment_id}: {doctor} - {patient_name} на {date_str} {time_str}")
                return True

        except Exception as e:
            print(f"❌ Ошибка при добавлении записи: {e}")
            return False

    def get_user_appointments(self, user_id: int) -> List[Dict[str, Any]]:
        """Получение всех активных записей пользователя"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM appointments 
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY date, time
                ''', (user_id,))
                rows = cursor.fetchall()
                appointments = [dict(row) for row in rows]
                print(f"📋 Найдено {len(appointments)} активных записей для пользователя {user_id}")
                return appointments
        except Exception as e:
            print(f"Ошибка получения записей пользователя: {e}")
            return []

    def cancel_appointment(self, user_id: int, appointment_id: Optional[int] = None) -> bool:
        """
        Отмена записи
        Если указан appointment_id - отменяет конкретную запись
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cancelled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if appointment_id is not None:
                    # Отменяем конкретную запись по ID
                    cursor.execute('''
                        UPDATE appointments 
                        SET status = 'cancelled', cancelled_at = ?
                        WHERE id = ? AND status = 'active'
                    ''', (cancelled_at, appointment_id))

                    if cursor.rowcount > 0:
                        conn.commit()
                        print(f"✅ Отменена запись #{appointment_id}")
                        return True
                    else:
                        print(f"❌ Запись #{appointment_id} не найдена или уже отменена")
                        return False
                else:
                    # Отменяем первую активную запись пользователя
                    cursor.execute('''
                        UPDATE appointments 
                        SET status = 'cancelled', cancelled_at = ?
                        WHERE user_id = ? AND status = 'active'
                        LIMIT 1
                    ''', (cancelled_at, user_id))

                    if cursor.rowcount > 0:
                        conn.commit()
                        print(f"✅ Отменена запись пользователя {user_id}")
                        return True
                    else:
                        print(f"❌ Нет активных записей для пользователя {user_id}")
                        return False

        except Exception as e:
            print(f"❌ Ошибка при отмене записи: {e}")
            return False

    def get_appointment_by_id(self, appointment_id: int) -> Optional[Dict[str, Any]]:
        """Получение записи по ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM appointments WHERE id = ?', (appointment_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            print(f"Ошибка получения записи: {e}")
            return None

    def get_available_times(self, doctor: str, date_str: str) -> List[str]:
        """Получение доступных временных слотов"""
        all_times = ["09:00", "10:00", "11:00", "12:00", "13:00",
                     "14:00", "15:00", "16:00", "17:00", "18:00", "19:00"]

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT time FROM appointments 
                    WHERE doctor = ? AND date = ? AND status = 'active'
                ''', (doctor, date_str))
                rows = cursor.fetchall()
                booked_times = [row['time'] for row in rows]

                available = [time for time in all_times if time not in booked_times]
                print(f"🔍 Доступное время для {doctor} на {date_str}: {available}")
                return available
        except Exception as e:
            print(f"Ошибка получения доступного времени: {e}")
            return all_times


# Создаем глобальный экземпляр
calendar_manager = CalendarManager()