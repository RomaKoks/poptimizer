"""Тесты для описания колонок."""
from datetime import datetime

import pytest

from poptimizer.data.adapters.html import description


def test_date_parser():
    """Парсер для дат."""
    assert description.date_parser("-") is None
    assert description.date_parser("30.11.2018 (рек.)") == datetime(2018, 11, 30)
    assert description.date_parser("19.07.2017") == datetime(2017, 7, 19)


def test_date_parser_us():
    """Парсер для дат в американском формате."""
    assert description.date_parser_us("-") is None
    assert description.date_parser_us("07/10/2019") == datetime(2019, 7, 10)


def test_div_parser():
    """Парсер для дивидендов."""
    assert description.div_parser("30,4") == pytest.approx(30.4)
    assert description.div_parser("66.8 (рек.)") == pytest.approx(66.8)
    assert description.div_parser("78,9 (прогноз)") == pytest.approx(78.9)
    assert description.div_parser("2 097") == pytest.approx(2097.0)
    assert description.div_parser("-") is None


def test_div_parser_us():
    """Парсер для дивидендов в долларах."""
    assert description.div_parser_us("$0.51") == pytest.approx(0.51)
    assert description.div_parser_us("-") is None
