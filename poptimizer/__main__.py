"""Запуск основных операций с помощью CLI:

- эволюции
- оптимизация
- проверка статуса дивидендов
"""
import typer

from poptimizer.data.views import div_status, listing
from poptimizer.evolve import Evolution
from poptimizer.portfolio import Optimizer, load_from_yaml


def evolve():
    """Run evolution."""
    ev = Evolution()
    date = listing.last_history_date()
    port = load_from_yaml(date)
    ev.evolve(port)


def dividends(ticker: str):
    """Get dividends status."""
    div_status.dividends_validation(ticker)


def optimize(date: str = typer.Argument(..., help="YYYY-MM-DD")):
    """Optimize portfolio."""
    port = load_from_yaml(date)
    opt = Optimizer(port)
    print(opt.portfolio)
    print(opt.metrics)
    print(opt)
    div_status.new_dividends(tuple(port.index[:-2]))


if __name__ == "__main__":
    app = typer.Typer(help="Run poptimizer subcommands.", add_completion=False)

    app.command()(evolve)
    app.command()(dividends)
    app.command()(optimize)

    app(prog_name="poptimizer")
