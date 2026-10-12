"""trutina: a typed decision model. One forward pass, calibrated probabilities for every option."""

from trutina.schema import Question, Request

__all__ = ["Question", "Request", "Trutina"]


def __getattr__(name):  # the engine pulls in MLX; import it only when asked for
    if name == "Trutina":
        from trutina.engine import Trutina

        return Trutina
    raise AttributeError(name)
