import sys

sys.path.append(".")
from regression_models import *


def fit_regression(model_caller):
    model = model_caller()
    model.fit_polyCoeff()

    return


def run_regression_nanofins():
    fit_regression(model_caller=multipoly_nanofins_6)

    return


def run_regression_nanocylinders():
    fit_regression(model_caller=multipoly_my_nanocylinders_12)

    return


if __name__ == "__main__":
    # run_regression_nanofins()
    run_regression_nanocylinders()
