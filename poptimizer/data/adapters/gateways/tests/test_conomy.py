"""Тесты для загрузки с https://www.conomy.ru/."""
import pandas as pd
import pytest
from pyppeteer import errors
from pyppeteer.page import Page

from poptimizer.data.adapters.gateways import conomy
from poptimizer.data.adapters.html import description, parser


@pytest.mark.asyncio
async def test_browser_close_page(mocker):
    """Браузер закрывает открываемые страницы."""
    br = conomy.Browser()

    async with br.get_page() as page:
        assert isinstance(page, Page)
        assert not page.isClosed()

    assert page.isClosed()  # noqa: WPS441


@pytest.mark.asyncio
async def test_browser_loads_once(mocker):
    """Браузер загружается однажды."""
    br = conomy.Browser()

    assert await br._load_browser() is await br._load_browser()


@pytest.mark.asyncio
async def test_load_ticker_page(mocker):
    """Переход на страницу с тикером."""
    fake_page = mocker.AsyncMock()
    fake_element = mocker.AsyncMock()
    fake_page.xpath.return_value = [fake_element]

    await conomy._load_ticker_page(fake_page, "TICKER")

    fake_page.goto.assert_called_once_with(conomy.SEARCH_URL)
    fake_page.xpath.assert_called_once_with(conomy.SEARCH_FIELD)
    fake_element.type.assert_called_once_with("TICKER")


@pytest.mark.asyncio
async def test_load_dividends_table(mocker):
    """Загрузка таблицы с тикером."""
    fake_page = mocker.AsyncMock()
    fake_element = mocker.AsyncMock()
    fake_page.xpath.return_value = [fake_element]

    await conomy._load_dividends_table(fake_page)

    fake_page.xpath.assert_called_once_with(conomy.DIVIDENDS_MENU)
    fake_element.click.assert_called_once_with()


@pytest.mark.asyncio
async def test_get_html(mocker):
    """Последовательный переход и загрузка html с дивидендами."""
    fake_browser = mocker.patch.object(conomy, "BROWSER")
    ctx_mng = fake_browser.get_page.return_value
    fake_page = ctx_mng.__aenter__.return_value  # noqa: WPS609
    fake_load_ticker_page = mocker.patch.object(conomy, "_load_ticker_page")
    mocker.patch.object(conomy, "_load_dividends_table")

    html = await conomy._get_html("UNAC")

    fake_load_ticker_page.assert_called_once_with(fake_page, "UNAC")
    assert html is fake_page.content.return_value


TICKER_CASES = (
    ("GAZP", True),
    ("SNGSP", False),
    ("WRONG", None),
    ("AAPL-RM", None),
)


@pytest.mark.parametrize("ticker, answer", TICKER_CASES)
def test_is_common(ticker, answer):
    """Проверка, что тикер соответствует обыкновенной акции."""
    if answer is None:
        with pytest.raises(description.ParserError, match="Некорректный тикер"):
            conomy._is_common(ticker)
    else:
        assert conomy._is_common(ticker) is answer


DESC_CASES = (
    ("CHMF", 7),
    ("SNGSP", 8),
)


@pytest.mark.parametrize("ticker, answer", DESC_CASES)
def test_get_col_desc(ticker, answer):
    """Правильное составление описания в зависимости от типа акции."""
    date, div = conomy._get_col_desc(ticker)
    assert date.num == 5
    assert div.num == answer


DF = pd.DataFrame(
    [[4.0], [1.0], [2.0], [None]],
    index=["2020-01-20", "2014-11-25", "2014-11-25", None],
    columns=["BELU"],
)
DF_REZ = pd.DataFrame(
    [[3.0], [4.0]],
    index=["2014-11-25", "2020-01-20"],
    columns=["BELU"],
)


@pytest.mark.asyncio
async def test_conomy_gateway(mocker):
    """Группировка и сортировка полученных данных."""
    mocker.patch.object(conomy, "_get_html")
    mocker.patch.object(conomy, "_get_col_desc")
    mocker.patch.object(parser, "get_df_from_html", return_value=DF)

    gateway = conomy.ConomyGateway()
    pd.testing.assert_frame_equal(await gateway.get("BELU"), DF_REZ)


@pytest.mark.asyncio
async def test_conomy_gateway_web_error(mocker):
    """Регрессионный тест на ошибку в загрузке данных."""
    mocker.patch.object(conomy, "_get_html", side_effect=errors.TimeoutError)

    gateway = conomy.ConomyGateway()
    df = await gateway.get("BELU")
    pd.testing.assert_frame_equal(df, pd.DataFrame(columns=["BELU"]))
