# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Type, Tuple

import kats.utils.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from kats.detectors import changepoint_evaluator
from kats.detectors import cusum_model
from kats.detectors.changepoint_evaluator import TuringEvaluator
from kats.detectors.detector import DetectorModel
from kats.detectors.detector_benchmark import (
    decompose_params,
    DetectorModelSet,
    SUPPORTED_METRICS,
)
from kats.utils.time_series_parameter_tuning import TimeSeriesParameterTuning
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.model.problem import Problem
from pymoo.model.result import Result
from pymoo.optimize import minimize

MINIMIZE = "minimize"
MAXIMIZE = "maximize"
OPTIMIZATION_GOAL_OPTIONS = {MINIMIZE, MAXIMIZE}


class HPT_Problem(Problem):
    """
    Multi-objective hyper parameter tuning problem.
    You can specify the objectives that you want optimize from the list of SUPPORTED_METRICS. For each objective you need to
    provide optimization goal (minimize or maximize). For example, if you want to minimize delay and maximize F1-score you
    could provide objectives_and_goals = {"f_score": "maximize", "delay": "minimize"}.
    You can also provide more than two objectives if you like.
    """

    def __init__(
        self,
        search_grid: TimeSeriesParameterTuning,
        data_df: pd.DataFrame,
        objectives_and_goals: Dict[str, str],
    ):
        self._validate_objectives_and_goals(objectives_and_goals)
        self.objectives_and_goals = objectives_and_goals

        # Make a list so that we always calculate fitness objectives in deterministic order.
        self.objectives = list(objectives_and_goals.keys())
        tunable_parameters = search_grid.get_search_space().tunable_parameters
        self.par_to_val = {}
        for par in tunable_parameters:
            self.par_to_val[par] = tunable_parameters[par].values

        # Make a list of the keys (tunable parameters) so that the order is deterministic.
        self.tunable_parameters = list(tunable_parameters.keys())
        self.lower_limits, self.upper_limits = self.get_upper_and_lower_limits()
        self.n_vars = len(tunable_parameters)
        self.all_solutions = {}
        self.data_df = data_df
        super().__init__(
            n_var=self.n_vars,
            n_obj=len(self.objectives),
            n_constr=0,  # Currently no constraints for the fitness objectives.
            xl=np.array(self.lower_limits),
            xu=np.array(self.upper_limits),
            # We solve an integer problem where each integer maps to hyper parameter.
            type_var=int,
            elementwise_evaluation=True,
        )
        self.turing_model = changepoint_evaluator.TuringEvaluator(
            is_detector_model=True, detector=cusum_model.CUSUMDetectorModel
        )

    def _validate_objectives_and_goals(self, objectives_and_goals: Dict[str, str]):
        self._check_if_all_valid(
            values_to_check=list(objectives_and_goals.keys()),
            expected_values=SUPPORTED_METRICS,
            explanation="Objectives",
        )

        self._check_if_all_valid(
            values_to_check=list(objectives_and_goals.values()),
            expected_values=OPTIMIZATION_GOAL_OPTIONS,
            explanation="Optimization goal",
        )

    def _check_if_all_valid(
        self, values_to_check: List[str], expected_values: set, explanation: str
    ):
        if not all(
            [value_to_check in expected_values for value_to_check in values_to_check]
        ):
            raise Exception(
                f"{explanation} must be listed in {expected_values}. You provided {values_to_check}."
            )

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["F"] = self.get_fitness(x)

    def get_fitness(self, x: np.ndarray):
        pars = self.decode_solution(x)
        params_model, threshold_low, threshold_high = decompose_params(pars)
        results = self.turing_model.evaluate(
            data=self.data_df,
            model_params=params_model,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )

        self.all_solutions[self.get_unique_id_for_solution(x)] = results
        fitness = [0] * self.n_obj
        averaged_results = np.mean(results)
        for i in range(self.n_obj):
            # For maximization problem, multiply the result by -1.
            fitness[i] = (
                averaged_results[self.objectives[i]]
                if self.objectives_and_goals[self.objectives[i]] == MINIMIZE
                else -averaged_results[self.objectives[i]]
            )

        return fitness

    def get_upper_and_lower_limits(self):
        upper_limits = []

        """
        We assign the limits in the order of tunable_parameters list. The order of that list will not
        change which is very important so that we can match the solution vector back to tunable parameters.
        """
        for key in self.par_to_val:
            upper_limits.append(len(self.par_to_val[key]) - 1)

        # All tunable_parameters should have at least one option.
        lower_limits = [0] * len(self.par_to_val)
        return lower_limits, upper_limits

    def decode_solution(self, x: np.ndarray) -> Dict[str, float]:
        pars = {}
        i = 0
        for key in self.tunable_parameters:
            pars[key] = self.par_to_val[key][x[i]]
            i += 1
        return pars

    def get_unique_id_for_solution(self, x: np.ndarray) -> str:
        return ",".join([str(x_component) for x_component in x])


class MultiObjectiveModelOptimizer(DetectorModelSet):
    def __init__(
        self,
        model_name: str,
        model: Type[DetectorModel],
        parameters_space: List[Dict],
        data_df: pd.DataFrame,
        n_gen: int,
        pop_size: int,
        objectives_and_goals: Dict[str, str],
    ):
        super().__init__(model_name, model)
        self.model_name = model_name
        self.model = model
        self.result = {}
        self.solutions = pd.DataFrame()
        self.parameters = parameters_space
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.hpt_problem = HPT_Problem(
            search_grid=tpt.SearchMethodFactory.create_search_method(
                parameters=self.parameters
            ),
            data_df=data_df,
            objectives_and_goals=objectives_and_goals,
        )

    # This overrides the base method
    def evaluate(
        self, data_df: pd.DataFrame
    ) -> Tuple[Dict[str, pd.DataFrame], TuringEvaluator]:

        logging.info("Creating multi-objective optimization problem.")
        method = get_algorithm(
            "nsga2",
            pop_size=self.pop_size,
            crossover=get_crossover(
                "int_sbx",
                prob=1.0,
                eta=3.0,
                prob_per_variable=(1 / self.hpt_problem.n_var),
            ),
            mutation=get_mutation("int_pm", eta=3.0),
            eliminate_duplicates=True,
            sampling=get_sampling("int_random"),
        )

        logging.info(
            "Running multi-objective optimization with pop_size {self.pop_size} and n_gen {self.n_gen}."
        )
        res = minimize(
            self.hpt_problem,
            method,
            ("n_gen", self.n_gen),
            verbose=True,
            seed=1,
            save_history=False,
        )

        self.get_results(res)
        self.get_hyper_parameters_and_results_for_non_dominated_solutions(res)
        logging.info("Multi-objective optimization completed.")
        return self.result, self.hpt_problem.turing_model

    def get_params(self):
        return self.solutions

    def get_results(self, res: Result):
        self.result = {}
        for id in range(len(res.X)):
            self.result[f"moo_solution_{id}"] = self.hpt_problem.all_solutions[
                self.hpt_problem.get_unique_id_for_solution(res.X[id])
            ]

    def get_hyper_parameters_and_results_for_non_dominated_solutions(self, res: Result):
        solutions = []
        for id in range(len(res.X)):
            decoded_solution = self.hpt_problem.decode_solution(res.X[id])
            uniq_id = self.hpt_problem.get_unique_id_for_solution(res.X[id])
            curr_solution_mean = np.mean(self.hpt_problem.all_solutions[uniq_id])
            decoded_solution["solution_id"] = id
            for metric in SUPPORTED_METRICS:
                decoded_solution[metric] = curr_solution_mean[metric]
            solutions.append(decoded_solution)
        self.solutions = pd.DataFrame(solutions)

    def update_benchmark_results(self, benchmark_results: Dict):
        benchmark_results.update(self.result)
