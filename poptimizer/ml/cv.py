"""Кросс-валидация и оптимизация гиперпараметров ML-модели."""
import functools
import logging

import catboost
import hyperopt
import numpy as np
from hyperopt import hp
from scipy import stats

from poptimizer.config import POptimizerError, ML_PARAMS, YEAR_IN_TRADING_DAYS
from poptimizer.ml.examples import Examples
from poptimizer.ml.feature import ON_OFF
from poptimizer.portfolio import Portfolio

# Базовые настройки catboost
MAX_ITERATIONS = 10000
SEED = 284_704
TECH_PARAMS = dict(
    loss_function="RMSE",
    eval_metric="RMSE:use_weights=True",
    iterations=MAX_ITERATIONS,
    random_state=SEED,
    od_type="Iter",
    od_wait=int(MAX_ITERATIONS ** 0.5),
    verbose=False,
    allow_writing_files=False,
)

# Настройки hyperopt
MAX_SEARCHES = 100

# Диапазоны поиска ключевых гиперпараметров относительно базового значения параметров
# Рекомендации Яндекс - https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/

# OneHot кодировка - учитывая количество акций в портфеле используется cat-кодировка или OneHot-кодировка
ONE_HOT_SIZE = [2, 1000]

# Диапазон поиска скорости обучения
LEARNING_RATE = [3.2e-03, 0.022]

# Диапазон поиска глубины деревьев
MAX_DEPTH = 16

# Диапазон поиска параметра L2-регуляризации
L2_LEAF_REG = [4.2e-01, 4.4]

# Диапазон поиска случайности разбиений
RANDOM_STRENGTH = [2.8e-01, 2.8]

# Диапазон поиска интенсивности бэггинга
BAGGING_TEMPERATURE = [0.36, 2.5e00]


def log_space(space_name: str, interval):
    """Создает логарифмическое вероятностное пространство"""
    lower, upper = interval
    lower, upper = np.log(lower), np.log(upper)
    return hp.loguniform(space_name, lower, upper)


def get_model_space() -> dict:
    """Создает вероятностное пространство для параметров регрессии."""
    space = {
        "one_hot_max_size": hp.choice("one_hot_max_size", ONE_HOT_SIZE),
        "learning_rate": log_space("learning_rate", LEARNING_RATE),
        "depth": hp.choice("depth", list(range(1, MAX_DEPTH + 1))),
        "l2_leaf_reg": log_space("l2_leaf_reg", L2_LEAF_REG),
        "random_strength": log_space("rand_strength", RANDOM_STRENGTH),
        "bagging_temperature": log_space("bagging_temperature", BAGGING_TEMPERATURE),
    }
    return space


def float_bounds_check(
    name, value, interval, bound: float = 0.1, increase: float = 0.2
):
    """Анализирует близость к границам интервала и предлагает расширить."""
    lower, upper = interval
    if value * (1 + bound) > upper:
        print(
            f"\nНеобходимо расширить {name} до [{lower}, {value * (1 + increase):0.1e}]"
        )
    elif value / (1 + bound) < lower:
        print(
            f"\nНеобходимо расширить {name} до [{value / (1 + increase):0.1e}, {upper}]"
        )


def check_model_bounds(params: dict, bound: float = 0.1, increase: float = 0.2):
    """Проверяет и дает рекомендации о расширении границ пространства поиска параметров.

    Для целочисленных параметров - предупреждение выдается на границе диапазона и рекомендуется
    расширить диапазон на 1.
    Для реальных параметров - предупреждение выдается в 10% от границе и рекомендуется расширить границы
    поиска на 10%, 20%.
    """
    names = ["learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature"]
    intervals = [LEARNING_RATE, L2_LEAF_REG, RANDOM_STRENGTH, BAGGING_TEMPERATURE]
    for name, interval in zip(names, intervals):
        value = params[name]
        float_bounds_check(name, value, interval, bound, increase)


def make_model_params(data_params: tuple, model_params: dict) -> dict:
    """Формирует параметры модели.

    Добавляет общие технические параметры и вставляет корректные данные по отключенным признакам.
    """
    result = dict(**TECH_PARAMS, **model_params)
    result["ignored_features"] = []
    num = 0
    for _, feat_params in data_params[1:]:
        periods = feat_params.get("periods", 1)
        if feat_params.get(ON_OFF, True) is False:
            result["ignored_features"].extend(list(range(num, num + periods)))
        num += periods
    return result


def incremental_return(r: np.array, r_expected: np.array, std_2: np.array) -> float:
    """Вычисляет доходность оптимального инкрементального портфеля.

    Оптимальный портфель сроится из допущения отсутствия корреляции. Размеры позиций нормируются для
    достижения фиксированного СКО портфеля.

    :param r:
        Фактическая доходность.
    :param r_expected:
        Прогнозная доходность.
    :param std_2:
        Прогнозный квадрат СКО.
    :return:
        Доходность нормированного по СКО оптимального инкрементального портфеля.
    """
    r_weighted = (r_expected / std_2).sum() / (1 / std_2).sum()
    weight = (r_expected - r_weighted) / std_2
    std_portfolio = (weight ** 2 * std_2).sum() ** 0.5
    weight = weight / std_portfolio
    return (r * weight).sum()


def t_of_cov(labels, labels_val, std_2, n_tickers):
    """Значимость доходов от предсказания.

    Если делать ставки на сигнал, пропорциональные разнице между сигналом и его матожиданием,
    то средний доход от таких ставок равен ковариации между сигналом и предсказываемой величиной.

    Каждый день мы делаем ставки на по всем имеющимся бумагам, соответственно можем получить оценки
    ковариации для отдельных дней и оценить t-статистику отличности ковариации (нашей прибыли) от нуля.
    """
    days, rez = divmod(len(labels_val), n_tickers)
    if rez:
        logging.info("Слишком длинные признаки и метки - сократи их длину!!!")
        return 0
    rs = np.zeros(days)
    for i in range(days):
        r = labels[i * n_tickers : (i + 1) * n_tickers]
        r_expected = labels_val[i * n_tickers : (i + 1) * n_tickers]
        std_2_ = std_2[i * n_tickers : (i + 1) * n_tickers]

        rs[i] = incremental_return(r, r_expected, std_2_)

    return rs.mean() / rs.std(ddof=1) * YEAR_IN_TRADING_DAYS ** 0.5


def valid_model(params: dict, examples: Examples, verbose=False) -> dict:
    """Осуществляет валидацию модели по R2.

    Осуществляется проверка, что не достигнут максимум итераций. Возвращается нормированное на СКО RMSE
    прогноза, R2-score, R и параметры данных и модели с оптимальным количеством итераций в формате
    целевой функции hyperopt.

    :param params:
        Словарь с параметрами модели и данных.
    :param examples:
        Класс создания обучающих примеров.
    :param verbose:
        Логировать ли параметры - используется при оптимизации гиперпараметров.
    :return:
        Словарь с результатом в формате hyperopt:

        * ключ 'loss' - нормированная RMSE на кросс-валидации (для hyperopt),
        * ключ 'status' - успешного прохождения (для hyperopt),
        * ключ 'params' - параметры данных.
        * ключ 'model' - параметры модели, в которые добавлено оптимальное количество итераций
        градиентного бустинга на кросс-валидации и общие настройки.
        * дополнительные ключи метрик качества модели.
    """
    if verbose:
        logging.info(f"Параметры модели:\n{params}")

    data_params, model_params = params["data"], params["model"]
    train_pool_params, val_pool_params = examples.train_val_pool_params(data_params)
    train_pool = catboost.Pool(**train_pool_params)
    if len(val_pool_params["data"]) == 0:
        return dict(loss=0, status=hyperopt.STATUS_OK)
    val_pool = catboost.Pool(**val_pool_params)
    n_tickers = len(examples.tickers)
    model_params = make_model_params(data_params, model_params)
    clf = catboost.CatBoostRegressor(**model_params)
    clf.fit(train_pool, eval_set=val_pool)

    if clf.tree_count_ == MAX_ITERATIONS:
        raise POptimizerError(f"Необходимо увеличить MAX_ITERATIONS = {MAX_ITERATIONS}")
    model_params["iterations"] = clf.tree_count_

    # RMSE нормируется на сумму весов, которые равны обратной величине квадрата СКО.
    # Для получения среднего отношения ошибки к СКО необходимо умножить на сумму весов и поделить на
    # их количество.
    scores = clf.get_best_score()["validation"]
    std = scores["RMSE"] * val_pool_params["weight"].mean() ** 0.5

    labels = val_pool_params["label"]
    labels_val = clf.predict(val_pool)
    r = stats.pearsonr(labels, labels_val)[0]
    r_rang = stats.spearmanr(labels, labels_val)[0]

    test_pool_params = examples.test_pool_params(data_params)
    labels = test_pool_params["label"]
    test_pool = catboost.Pool(**test_pool_params)
    labels_test = clf.predict(test_pool)
    std_2 = 1 / test_pool_params["weight"]

    t = t_of_cov(labels, labels_test, std_2, n_tickers)

    if verbose:
        logging.info(f"R: {r}\nt: {t}\n")
    return dict(
        loss=-t,
        status=hyperopt.STATUS_OK,
        std=std,
        r=r,
        r_rang=r_rang,
        t=t,
        data=data_params,
        model=model_params,
    )


def optimize_hyper(examples: Examples) -> tuple:
    """Ищет и  возвращает лучший набор гиперпараметров без количества итераций.

    :param examples:
        Класс генератора примеров для модели.
    :return:
        Оптимальные параметры модели.
    """
    objective = functools.partial(valid_model, examples=examples, verbose=True)
    param_space = dict(data=examples.get_params_space(), model=get_model_space())
    best = hyperopt.fmin(
        objective,
        space=param_space,
        algo=hyperopt.tpe.suggest,
        max_evals=MAX_SEARCHES,
        rstate=np.random.RandomState(SEED),
        show_progressbar=False,
    )
    # Преобразование из внутреннего представление в исходное пространство
    best_params = hyperopt.space_eval(param_space, best)
    check_model_bounds(best_params["model"])
    return best_params


def print_result(name, params, examples: Examples):
    """Проводит кросс-валидацию, выводит ее основные метрики."""
    cv_results = valid_model(params, examples)
    print(
        f"\n{name}"
        f"\nR - {cv_results['r']:0.4%}"
        f"\nR_rang - {cv_results['r_rang']:0.4%}"
        f"\nT - {cv_results['t']:0.4}"
        f"\nКоличество итераций - {cv_results['model']['iterations']}"
        f"\n{cv_results['data']}"
        f"\n{cv_results['model']}"
    )
    return cv_results["loss"]


def find_better_model(port: Portfolio, params: dict = ML_PARAMS):
    """Ищет оптимальную модель и сравнивает с базовой - результаты сравнения распечатываются."""
    examples = Examples(tuple(port.index[:-2]), port.date, params["data"])
    print("\nИдет поиск новой модели")
    new_params = optimize_hyper(examples)
    base_params = ML_PARAMS
    base_name = "Базовая модель"
    base_loss = print_result(base_name, base_params, examples)
    new_name = "Найденная модель"
    new_loss = print_result("Найденная модель", new_params, examples)

    if base_loss < new_loss:
        print(f"\nЛУЧШАЯ МОДЕЛЬ - {base_name}" f"\n{base_params}")
    else:
        print(f"\nЛУЧШАЯ МОДЕЛЬ - {new_name}" f"\n{new_params}")
