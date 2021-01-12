"""Оптимизатор портфеля."""
import itertools
import math
from typing import Tuple
from datetime import datetime
import pandas as pd
from scipy import stats

from poptimizer import config
from poptimizer.config import MAX_TRADE
from poptimizer.dl.features.data_params import FORECAST_DAYS
from poptimizer.portfolio import metrics
from poptimizer.portfolio.portfolio import Portfolio, CASH, PORTFOLIO

# На сколько сделок разбивается операция по покупке/продаже акций
TRADES = 1
# Значимость отклонения градиента от нуля
P_VALUE = 0.05

# Издержки в годовом выражении для двух операций
COSTS = (config.YEAR_IN_TRADING_DAYS * 2 / FORECAST_DAYS) * (0.025 / 100)


class Optimizer:
    """Предлагает сделки для улучшения метрики портфеля."""

    def __init__(self, portfolio: Portfolio, p_value: float = P_VALUE):
        """Учитывается градиент, его ошибку и ликвидность бумаг.

        :param portfolio:
            Оптимизируемый портфель.
        :param p_value:
            Требуемая значимость отклонения градиента от нуля.
        """
        self._portfolio = portfolio
        self._p_value = p_value
        self._metrics = metrics.MetricsResample(portfolio)

    def __str__(self) -> str:
        blocks = [
            "\nОПТИМИЗАЦИЯ ПОРТФЕЛЯ",
            self._p_value_block(),
            f"{self.best_combination}",
        ]
        return "\n\n".join(blocks)

    def _p_value_block(self) -> str:
        """Информация значимости при множественном тестировании гипотез."""
        blocks = [
            "Оценка значимости:",
            f"forecasts = {self.n_forecasts}",
            f"p-value = {self.p_value:.2%}",
            f"trials = {self.trials}",
        ]
        return "\n".join(blocks)

    @property
    def portfolio(self) -> Portfolio:
        """Оптимизируемый портфель."""
        return self._portfolio

    @property
    def metrics(self) -> metrics.MetricsResample:
        """Метрики портфеля."""
        return self._metrics

    @property
    def p_value(self) -> float:
        """Уровень значимости отклонения градиента от нуля."""
        return self._p_value

    @property
    def n_forecasts(self) -> int:
        """Количество прогнозов."""
        return self.metrics.count

    @property
    def trials(self) -> int:
        """Количество тестов на значимость.

        Продать можно позиции с не нулевым весом.

        Можно уйти в кэш или купить любую позицию, кроме продаваемой и с нулевым фактором объема.
        Для позиции с нулевым фактором объема - уйти в кеш или купить любую позицию кроме себя.
        """
        positions_to_sell = (self.portfolio.shares[:-2] > 0).sum()
        positions = len(self.portfolio.shares) - 2
        return positions_to_sell * positions - positions_to_sell + 1

    @property
    def best_combination(self):
        """Лучшие комбинации для торговли.

        Для каждого актива, который можно продать со значимым улучшением выбирается актив с
        максимальной вероятностью улучшения.
        """
        rez = self._wilcoxon_tests()

        rez = pd.DataFrame(list(rez), columns=["SELL", "BUY", "GRAD_DIFF", "TURNOVER", "P_VALUE"])
        rez["SORT"] = (pd.qcut(rez["GRAD_DIFF"], min(rez.shape[0], 10), labels=False) + 1) \
                      * (pd.qcut(rez["TURNOVER"], min(rez.shape[0], 3), labels=False) + 1) \
                      / (pd.qcut(rez['P_VALUE'], min(rez.shape[0], 5), labels=False) + 1)

        rez = rez.sort_values("SORT", ascending=False)
        rez.drop("SORT", axis=1)
        rez.index = pd.RangeIndex(start=1, stop=len(rez) + 1)

        return self._add_sell_buy_quantity(rez)

    def _wilcoxon_tests(self) -> Tuple[str, str, float, float]:
        """Осуществляет тестирование всех допустимых пар активов с помощью теста Вилкоксона.

        Возвращает все значимо улучшающие варианты сделок в формате:

        - Продаваемый тикер
        - Покупаемый тикер
        - Медиана разницы в градиенте
        - Значимость скорректированная на общее количество тестов и оборачиваемость покупаемого тикера.
        """
        positions_to_sell = self.portfolio.index[:-2][self.portfolio.shares[:-2] > 0]
        positions_with_cash = self.portfolio.index[:-1]
        all_gradients = self.metrics.all_gradients
        trials = self.trials
        turnover_all = self.portfolio.turnover_factor
        for sell, buy in itertools.product(positions_to_sell, positions_with_cash):
            if sell == buy or turnover_all[buy] == 0:
                continue

            diff = all_gradients.loc[buy] - all_gradients.loc[sell] - COSTS
            _, alfa = stats.wilcoxon(diff, alternative="greater", correction=True)

            alfa *= trials

            if alfa < P_VALUE:
                yield [
                    sell,
                    buy,
                    diff.median(),
                    min(turnover_all[buy], turnover_all[sell]),
                    alfa,
                ]

    def _add_sell_buy_quantity(self, rez: pd.DataFrame) -> pd.DataFrame:
        """Добавляет колонки с объемами покупки и продажи.

        Объем продажи и покупки делится на TRADES операций.

        Объем продажи не может быть больше имеющегося количества и не должен приводить к повышению
        кэша до MAX_TRADE от размера портфеля.

        Объем покупки равен количеству имеющегося кеша.
        """
        portfolio = self.portfolio
        lot_value_per_trades = (portfolio.price * portfolio.lot_size) * TRADES
        value = portfolio.value

        max_trade = value[PORTFOLIO] * MAX_TRADE
        max_trade = value.apply(lambda x: min(x, max(0, max_trade - value[CASH])))
        sell_size = max_trade / lot_value_per_trades
        sell_size = sell_size.apply(lambda x: math.ceil(x))
        rez["Q_SELL"] = 0
        rez["Q_SELL"] = rez["SELL"].apply(lambda ticker: sell_size[ticker])

        buy_size = value[CASH] / lot_value_per_trades
        buy_size = buy_size.apply(lambda x: math.ceil(max(0, x)))
        rez["Q_BUY"] = 0
        rez["Q_BUY"] = rez["BUY"].apply(lambda ticker: buy_size[ticker])

        def save_to_excel(filename, dfs):
            # Given a dict of dataframes, for example:
            # dfs = {'gadgets': df_gadgets, 'widgets': df_widgets}
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            for sheetname, df in dfs.items():  # loop through `dict` of dataframes
                df.to_excel(writer, sheet_name=sheetname)  # send df to writer
                worksheet = writer.sheets[sheetname]  # pull worksheet object
                for idx, col in enumerate(df.columns):  # loop through all columns
                    series = df[col]
                    max_len = series.astype(str).map(len).max() + 1
                    worksheet.set_column(idx + 1, idx + 1, max_len)  # set column width
            writer.save()

        sel = rez.groupby('SELL')['SORT'].mean().sort_values(ascending=False).to_frame(name='SORT')
        sel['SORT'] = sel['SORT'] / sel['SORT'].sum()
        buy = rez.groupby('BUY')['SORT'].mean().sort_values(ascending=False).to_frame(name='SORT')
        buy['SORT'] = buy['SORT'] / buy['SORT'].sum()
        save_to_excel(f'portfolio/reports/rec_ops_{str(datetime.today())[:10]}.xlsx',
                      {'rating_full': rez,
                       'SELL': sel,
                       'BUY': buy})

        return rez[["SELL", "Q_SELL", "BUY", "Q_BUY", "GRAD_DIFF", "TURNOVER", "P_VALUE"]]
