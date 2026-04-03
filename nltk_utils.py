"""Утилиты для обработки естественного языка"""
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Загрузка данных NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence: str) -> list:
    """
    Разбивает предложение на токены/слова
    """
    return nltk.word_tokenize(sentence)

def stem(word: str) -> str:
    """
    Стемминг - приведение слова к корневой форме
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence: list, words: list) -> np.ndarray:
    """
    Создает мешок слов (bag of words)

    Args:
        tokenized_sentence: токенизированное предложение
        words: словарь всех слов

    Returns:
        numpy массив с бинарными признаками (0 или 1 для каждого слова)
    """
    # Стемминг слов в предложении
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Создаем мешок слов
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
