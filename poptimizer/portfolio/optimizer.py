"""Оптимизатор портфеля."""
import functools
import itertools
from datetime import datetime
import pandas as pd
from scipy import stats
from sklearn.preprocessing import quantile_transform

from poptimizer import config
from poptimizer.portfolio import metrics
from poptimizer.portfolio.portfolio import CASH, PORTFOLIO, Portfolio


def save_to_excel(filename, dfs):
    # Given a dict of dataframes, for example:
    # dfs = {'gadgets': df_gadgets, 'widgets': df_widgets}
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheetname, df in dfs.items():  # loop through `dict` of dataframes
        df.to_excel(writer, sheet_name=sheetname)  # send df to writer
        worksheet = writer.sheets[sheetname]  # pull worksheet object
        for idx, col in enumerate(df.columns):  # loop through all columns
            series = df[col]
            max_len = max(len(col), series.astype(str).map(len).max()) + 1
            worksheet.set_column(idx + 1, idx + 1, max_len)  # set column width
    writer.save()


class Optimizer:
    """Предлагает сделки для улучшения метрики портфеля."""

    def __init__(self, portfolio: Portfolio, p_value: float = config.P_VALUE):
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
        df = self.best_combination()
        blocks = [
            "\nОПТИМИЗАЦИЯ ПОРТФЕЛЯ",
            f"forecasts = {self.metrics.count}",
            f"p-value = {self._p_value:.2%}",
            f"trials = {self.trials}",
            f"match = {len(df)}",
            f"for sale = {len(df['SELL'].unique())}",
            f"\n{df}",
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

    @functools.cached_property
    def trials(self) -> int:
        """Количество тестов на значимость."""
        return sum(1 for _ in self._acceptable_trades())

    def _calculate_lots_to_buy_sell(self, rez):
        recomendations = {}
        cash_threshold = self.portfolio.value['PORTFOLIO'] * config.MAX_TRADE
        # cash_threshold = min(self.portfolio.value['CASH'], cash_threshold)
        for action in ['BUY', 'SELL']:
            banned = set()
            zero_lots_exists = True
            rec = None
            while zero_lots_exists:
                temp = rez.loc[~(rez[action].isin(banned))].copy()
                # преобразуем Q_H_MEAN в пропорцию от общего бюджета
                rec = (temp.groupby(action)['Q_H_MEAN'].sum() / temp['Q_H_MEAN'].sum()).to_frame(name='proportion')
                rec['lot_price'] = (self.portfolio.lot_size.loc[rec.index] * self.portfolio.price.loc[rec.index]).round(2)
                # вычисляем по пропорции конкретную сумму по тикеру
                rec['SUM'] = rec['proportion'] * cash_threshold
                # считаем ближацшее к ней целое количество лотов
                rec['lots'] = (rec['SUM'] / rec['lot_price']).round().astype(int)
                # проверяем есть ли тикеры с 0м количеством лотов и последовательно добавляем в бан тикеры
                # начиная с конца, то есть с те, у которых меньшая пропорция, до тех пор, пока не останутся
                # только тикеры с ненулевым количеством лотов.
                zero_lots_exists = (rec['lots'] < 1).any()
                rec.sort_values('proportion', inplace=True, ascending=False)
                if rec.shape[0] <= 1:
                    break
                banned.add(rec.index[-1])
            # корректируем сумму учитывая целое количество лотов
            rec['SUM'] = (rec['lots'] * rec['lot_price']).round(2)
            rec['proportion'] = rec['proportion'].round(3)
            rec['SHARES'] = rec['lots'] * self.portfolio.lot_size.loc[rec.index]
            rec = rec[['lot_price', 'lots', 'SHARES', 'SUM', 'proportion']]
            recomendations[action] = rec
        return recomendations

    def best_combination(self) -> pd.DataFrame:
        """Лучшие комбинации для торговли.

        Для каждого актива, который можно продать со значимым улучшением выбирается актив с
        максимальной вероятностью улучшения.
        """
        rez = pd.DataFrame(
            list(self._wilcoxon_tests()),
            columns=[
                "SELL",
                "BUY",
                "SML_DIFF",
                "B_DIFF",
                "R_DIFF",
                "TURNOVER",
                "P_VALUE",
            ],
        )
        if rez.shape[0] > 0:
            tmp = rez[['SML_DIFF', 'R_DIFF']].copy()
            rez['Q_H_MEAN'] = stats.hmean(quantile_transform(tmp), axis=1)    # гармоническое среднее квантилей (аналог F1)
            rez.sort_values(["Q_H_MEAN"], ascending=[False], inplace=True)
            lots = self._calculate_lots_to_buy_sell(rez)

            save_to_excel(f'portfolio/reports/rec_ops_{str(datetime.today())[:10]}.xlsx',
                          {'options': rez, 'lots_to_buy': lots['BUY'], 'lots_to_sell': lots['SELL']})
            rez.sort_values(["SML_DIFF"], ascending=[False], inplace=True)
            rez.index = pd.RangeIndex(start=1, stop=len(rez) + 1)
        return rez

    def _acceptable_trades(self) -> tuple[str, str, float]:
        positions = self.portfolio.index[:-2]
        weight = self.portfolio.weight
        turnover = self.portfolio.turnover_factor

        for sell, buy in itertools.product(positions, positions):
            if sell == buy:
                continue

            if weight[sell] == 0:
                continue

            factor = turnover[buy] - (weight[sell] + weight[CASH])
            if factor < 0:
                continue

            yield sell, buy, factor

    def _wilcoxon_tests(self) -> tuple[str, str, float, float, float, float]:
        """Осуществляет тестирование всех допустимых пар активов с помощью теста Вилкоксона."""
        all_gradients = self.metrics.all_gradients
        means = self.metrics.mean
        betas = self.metrics.beta

        for sell, buy, factor in self._acceptable_trades():
            mean = means[buy] - means[sell] - config.COSTS
            if _bad_mean(mean, means[PORTFOLIO]):
                continue

            diff = all_gradients.loc[buy] - all_gradients.loc[sell] - config.COSTS
            _, alfa = stats.wilcoxon(diff, alternative="greater", correction=True)
            alfa *= self.trials

            if alfa < self._p_value:
                yield [
                    sell,
                    buy,
                    diff.median(),
                    betas[sell] - betas[buy],
                    mean,
                    factor,
                    alfa,
                ]


def _bad_mean(mean: float, port_mean: float) -> bool:
    if config.MIN_RETURN is None:
        return False

    if port_mean > config.MIN_RETURN:
        return False

    return mean < 0
