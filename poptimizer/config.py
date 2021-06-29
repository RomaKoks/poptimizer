"""Основные настраиваемые параметры"""
import logging
import pathlib
import sys

import pandas as pd
import torch


class POptimizerError(Exception):
    """Базовое исключение."""


# Конфигурация логгера
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)])

# Устройство на котором будет производиться обучение
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #

# Количество колонок в распечатках без переноса на несколько страниц
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 180)
pd.set_option("display.width", None)

# Путь к директории с отчетами
REPORTS_PATH = pathlib.Path(__file__).parents[1] / "reports"

# Путь к директории с портфелями
PORT_PATH = pathlib.Path(__file__).parents[1] / "portfolio"

# Количество торговых дней в году
YEAR_IN_TRADING_DAYS = 12 * 21

# Ограничение на размер оборота — используется для предложения новых бумаг для анализа
MAX_TRADE = 1 / 130

# Максимальная популяция
MAX_POPULATION = 60

# Длинна прогноза в торговых днях
FORECAST_DAYS = 33

# Требуемая доходность, если None не используется
MIN_RETURN = None

# Значимость отклонения градиента от нуля
P_VALUE = 0.05

# Транзакционные издержки на две сделки
COSTS = (YEAR_IN_TRADING_DAYS / FORECAST_DAYS) * (0.025 / 100) * 2
