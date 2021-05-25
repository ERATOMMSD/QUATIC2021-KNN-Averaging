import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance

from numpy.linalg import norm

from pymoo.model.problem import Problem

class KNNMixin(object):
    """Base mixin for KNN work. Provides saving of history and some data parts."""
    def __init__(self, settings, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settings = settings
        self.generation = 0
        self.history = None

        assert len(self.settings["standard_deviation"]) == self.n_obj, "Number of objectives and standard deviation values need to be equal"

    def problem_name(self):
        return self.__class__.__bases__[1].__name__

    def name(self):
        return "_".join([base.__name__ for base in self.__class__.__bases__])


    def get_history_df(self):
        return pd.DataFrame(self.history, columns=self._get_column_headers())

    def store_history_to_file(self):
        self.get_history_df().to_csv(os.path.join(self.settings["output_folder"], "objectives.csv"), index=False)

    def _get_column_headers(self):
        return ["generation"] + \
               [f'x{idx}' for idx in range(self.n_var)] + \
               [f'calculated_y{idx}' for idx in range(self.n_obj)] + \
               [f'noisy_y{idx}' for idx in range(self.n_obj)] + \
               [f'objective_y{idx}' for idx in range(self.n_obj)] + \
               ['average of']


class KNNAvgMixin(KNNMixin):
    """
    This class does two things:
    1. Add noise to default problem evaluation
    2. Perform Std Euclidean Distance Averaging
    """
    def _evaluate(self, x, out, *args, **kwargs) -> None:

        # Here come the non-noisy values
        super()._evaluate(x, out, *args, **kwargs)

        calculated_Y = out["F"]

        """ Calculate Noise """
        self.generation += 1
        noise = np.column_stack(
            [np.random.normal(0, std_dev, len(calculated_Y)) for std_dev in self.settings["standard_deviation"]]
        )
        noisy_Y = calculated_Y + noise


        """ Calculate Objective_Y """
        objective_Y = noisy_Y.copy()
        candidates_for_avg = [1] * len(noisy_Y)

        # Temporary full history (so we can find KNN of history and current set)
        # Log Format is [Generation, X1, X2, ..., Xn, calc_Y1, ..., calc_Yn, noise_Y1, ..., noise_Yn, objective_Y1, ..., objective Yn]
        tmp_log = np.column_stack([[self.generation] * len(x), x, calculated_Y, noisy_Y, objective_Y, candidates_for_avg])
        tmp_full_log = tmp_log.copy()
        if self.history is not None:
            tmp_full_log = np.append(self.history, tmp_log, 0)

        KNN = self.settings["KNN"]
        if KNN != 1:
            assert KNN is None or KNN > 1

            # Calculate Std Euclidean Distances
            # Rows are distances of a tmp_log entry to the (10/20) values in tmp_log
            D = distance.cdist(tmp_full_log[:, 1:1 + (self.n_var)], tmp_log[:, 1:1 + self.n_var], 'seuclidean')

            for rowidx, row in enumerate(tmp_log):
                tmp_df = pd.DataFrame(tmp_full_log, columns=self._get_column_headers())
                tmp_df["StdEuclDist"] = D[:,rowidx]

                if self.settings["max_distance"] is not None:
                    tmp_df = tmp_df[tmp_df["StdEuclDist"] <= self.settings["max_distance"]]

                tmp_df = tmp_df.sort_values(by="StdEuclDist").head(KNN)  # sort and select only the first few ones
                tmp_df["weights"] = tmp_df["StdEuclDist"] * -1 + self.settings["max_distance"]  # linear weights !

                if self.settings["distance_weight"] == "squared":
                    tmp_df["weights"] = tmp_df["weights"] ** 2
                elif self.settings["distance_weight"] == "uniform":
                    tmp_df["weights"] = 1

                noisy_columns = [c for c in tmp_df.columns if "noisy_y" in c]
                if len(tmp_df) > 1:
                    objective_Y[rowidx] = np.average(tmp_df[noisy_columns], weights=tmp_df["weights"], axis=0)
                    candidates_for_avg[rowidx] = len(tmp_df)
                    pass
                else:
                    pass  # nothing to do here

        log = np.column_stack([[self.generation] * len(x), x, calculated_Y, noisy_Y, objective_Y, candidates_for_avg])

        """ Save the history -- including all three objective sets (no noise, noisy, averaged)"""
        if self.history is None:
            self.history = log
        else:
            self.history = np.append(self.history, log, 0)
        out["F"] = objective_Y


def wrap_problem(wrapper, problem, *args, **kwargs):
    """ This function dynamically wraps a problem inside a wrapper."""
    clazz = type(f"{wrapper.__name__}_{problem.__name__}", (wrapper, problem), dict())
    return clazz(*args, **kwargs)
