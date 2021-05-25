import os
import json
import datetime

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from pymoo.problems.multi import zdt

import knn_wrapper


def run_with_settings(settings, verbose=False, save=True, show=False, problem_vars=2):
    """Takes a setting, creates a wrapped problem and triggers pymoo's NSGA-II"""
    # Create the problem
    problem = knn_wrapper.wrap_problem(knn_wrapper.KNNAvgMixin, zdt.ZDT1, settings, n_var=problem_vars)

    # define a few settings
    settings["problem"] = problem.name()
    settings["base problem"] = problem.problem_name()
    settings["num vars"] = problem.n_var
    settings["num obj"] = problem.n_obj

    if save:
        # create the output folder
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        folder_name = f"{problem.name()}_NVar{problem.n_var}/{timestamp}"
        settings["output_folder"] = os.path.join("output", folder_name)
        print("Output goes to ", settings["output_folder"], "Settings:", settings)
        os.makedirs(settings["output_folder"], exist_ok=True)

        # store the configuration in a json file
        with open(os.path.join(settings["output_folder"], "config.json"), "w") as config_file:
            json.dump(settings, config_file, indent=4)
    else:
        print("Settings:", settings)

    algorithm = NSGA2(
        pop_size=settings["gen_size"],
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", prob=1.0, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   termination=get_termination("n_gen", settings["total_generations"]),
                   save_history=False,
                   verbose=verbose)

    # get the pareto-set and pareto-front for plotting
    ps = problem.pareto_set(use_cache=False)
    pf = problem.pareto_front(use_cache=False)

    if save:  # remember the past, write to file
        problem.store_history_to_file()

    """ Below is some Pymoo code for plotting output"""

    # Solution Space
    plot = Scatter(title="Solution Space", axis_labels="x")
    plot.add(res.X, s=30, facecolors='none', edgecolors='r')
    if ps is not None:
        plot.add(ps, plot_type="line", color="black", alpha=0.7)
    plot.do()
    if save:
        plot.save(os.path.join(settings["output_folder"], "SolutionSpace.png"))
    if show:
        plot.show()

    # Objective Space
    plot = Scatter(title="Objective Space")
    plot.add(res.F)
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.do()
    if save:
        plot.save(os.path.join(settings["output_folder"], "ObjectiveSpace.png"))
    if show:
        plot.show()


def create_settings_and_run():
    """Specifies the settings and calls the runner."""
    problem_vars = 4
    settings = dict(
        standard_deviation=[0.25, 0.25],  # the noises for each dimension (absolute values, in the paper this is the same in all dimensions!)
        KNN=10,  # how many neighbors to take into account
        gen_size=10,  # population size of the NSGA-II algorithms
        total_generations=100,  # how many generations to execute for
        max_distance=1,  # how far away is the zero weight
        distance_weight="squared",  # uniform, linear or squared
    )

    run_with_settings(settings, verbose=True, save=True, show=False, problem_vars=problem_vars)


if __name__ == "__main__":
    create_settings_and_run()