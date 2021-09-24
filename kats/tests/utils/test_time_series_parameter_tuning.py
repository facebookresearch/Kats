# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import kats.utils.time_series_parameter_tuning as tpt
import numpy as np
import pandas as pd
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
from ax.models.random.sobol import SobolGenerator
from ax.models.random.uniform import UniformGenerator
from kats.consts import SearchMethodEnum
from kats.models.arima import ARIMAModel
from kats.models.prophet import ProphetModel


class GridSearchTest(TestCase):
    def test_time_series_parameter_tuning_arima(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def arima_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=ARIMAModel.get_parameter_search_space(),
            selected_search_method=SearchMethodEnum.GRID_SEARCH,
        )
        time_series_parameter_tuner.generate_evaluate_new_parameter_values(
            evaluation_function=arima_evaluation_function
        )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 50)

    def test_time_series_parameter_tuning_prophet(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def prophet_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=ProphetModel.get_parameter_search_space(),
            selected_search_method=SearchMethodEnum.GRID_SEARCH,
        )
        time_series_parameter_tuner.generate_evaluate_new_parameter_values(
            evaluation_function=prophet_evaluation_function
        )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 25600)

    def test_grid_search_arm_count(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        params_in_json = [
            {
                "name": "x1",
                "type": "choice",
                "values": list(range(10)),
                "value_type": "int",
            },
            {
                "name": "x2",
                "type": "choice",
                "values": list(range(10)),
                "value_type": "int",
            },
            {
                "name": "x3",
                "type": "choice",
                "values": list(range(10)),
                "value_type": "int",
            },
            {
                "name": "x4",
                "type": "choice",
                "values": list(range(10)),
                "value_type": "int",
            },
        ]
        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=params_in_json,
            selected_search_method=SearchMethodEnum.GRID_SEARCH,
        )
        time_series_parameter_tuner.generate_evaluate_new_parameter_values(
            evaluation_function=evaluation_function
        )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 10000)

    def test_validate_parameters_format(self) -> None:
        parameters = [
            {"name": "test_param1", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "test_param2", "type": "choice", "values": [1, 2, 3]},
        ]
        tpt.TimeSeriesParameterTuning.validate_parameters_format(parameters)
        self.assertRaises(
            TypeError, tpt.TimeSeriesParameterTuning.validate_parameters_format, None
        )
        self.assertRaises(
            TypeError, tpt.TimeSeriesParameterTuning.validate_parameters_format, set()
        )
        self.assertRaises(
            ValueError, tpt.TimeSeriesParameterTuning.validate_parameters_format, []
        )
        self.assertRaises(
            ValueError, tpt.TimeSeriesParameterTuning.validate_parameters_format, [{}]
        )
        self.assertRaises(
            ValueError,
            tpt.TimeSeriesParameterTuning.validate_parameters_format,
            [{"key": "value"}, {}],
        )

    def test_custom_parameter_search_space(self) -> None:
        parameters = [
            {
                "name": "test_param1",
                "type": "choice",
                "values": [1, 2, 3],
                "is_ordered": True,
            },
            {
                "name": "test_param2",
                "type": "choice",
                "values": ["red", "green", "blue"],
            },
            {"name": "test_param3", "type": "fixed", "value": 4},
        ]
        time_series_parameter_tuning = tpt.GridSearch(parameters=parameters)
        self.assertIsInstance(
            time_series_parameter_tuning.parameters[0], ChoiceParameter
        )
        self.assertEqual(time_series_parameter_tuning.parameters[0].name, "test_param1")
        self.assertEqual(
            time_series_parameter_tuning.parameters[0].parameter_type, ParameterType.INT
        )
        self.assertEqual(time_series_parameter_tuning.parameters[0].values, [1, 2, 3])
        self.assertIsInstance(
            time_series_parameter_tuning.parameters[1], ChoiceParameter
        )
        self.assertEqual(time_series_parameter_tuning.parameters[1].name, "test_param2")
        self.assertEqual(
            time_series_parameter_tuning.parameters[1].parameter_type,
            ParameterType.STRING,
        )
        self.assertEqual(
            time_series_parameter_tuning.parameters[1].values, ["red", "green", "blue"]
        )
        self.assertIsInstance(
            time_series_parameter_tuning.parameters[2], FixedParameter
        )
        self.assertEqual(time_series_parameter_tuning.parameters[2].name, "test_param3")
        self.assertEqual(
            time_series_parameter_tuning.parameters[2].parameter_type, ParameterType.INT
        )
        self.assertEqual(time_series_parameter_tuning.parameters[2].value, 4)

    def test_time_series_parameter_tuning_arima_uniform_random_search(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def arima_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=ARIMAModel.get_parameter_search_space(),
            selected_search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        )
        self.assertIsInstance(
            # pyre-fixme[16]: `TimeSeriesParameterTuning` has no attribute
            #  `_random_strategy_model`.
            time_series_parameter_tuner._random_strategy_model.model,
            UniformGenerator,
        )
        for _ in range(3):
            time_series_parameter_tuner.generate_evaluate_new_parameter_values(
                evaluation_function=arima_evaluation_function, arm_count=4
            )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 12)

    def test_time_series_parameter_tuning_prophet_sobol_random_search(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def prophet_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=ProphetModel.get_parameter_search_space(),
            selected_search_method=SearchMethodEnum.RANDOM_SEARCH_SOBOL,
        )
        self.assertIsInstance(
            # pyre-fixme[16]: `TimeSeriesParameterTuning` has no attribute
            #  `_random_strategy_model`.
            time_series_parameter_tuner._random_strategy_model.model,
            SobolGenerator,
        )
        for _ in range(4):
            time_series_parameter_tuner.generate_evaluate_new_parameter_values(
                evaluation_function=prophet_evaluation_function, arm_count=5
            )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 20)

    def test_time_series_parameter_tuning_prophet_bayes_opt(self) -> None:
        random_state = np.random.RandomState(seed=0)

        def prophet_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        time_series_parameter_tuner = tpt.SearchMethodFactory.create_search_method(
            parameters=ProphetModel.get_parameter_search_space(),
            selected_search_method=SearchMethodEnum.BAYES_OPT,
            evaluation_function=prophet_evaluation_function,
            # objective_name='some_objective'
        )

        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )

        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 5)
        for _ in range(5):
            time_series_parameter_tuner.generate_evaluate_new_parameter_values(
                evaluation_function=prophet_evaluation_function, arm_count=1
            )
        parameter_values_with_scores = (
            time_series_parameter_tuner.list_parameter_value_scores()
        )
        # print(f'* * * {parameter_values_with_scores.to_string()}')
        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        self.assertEqual(len(parameter_values_with_scores.index), 10)

    # def test_outcome_constraint_without_filter(self) -> None:
    #     def run_model(x):
    #         precision = x
    #         recall = -x + 1
    #         return recall, precision

    #     def evaluate_recall_precision(params: Dict[str, float]) -> Dict[str, Tuple]:
    #         recall, precision = run_model(params["x"])
    #         return {"recall": (recall, 0.0), "precision": (precision, 0.0)}

    #     search_method = tpt.SearchMethodFactory.create_search_method(
    #         parameters=[
    #             {
    #                 "name": "x",
    #                 "type": "range",
    #                 "value_type": "float",
    #                 "bounds": [0.0, 1.0],
    #             }
    #         ],
    #         selected_search_method=SearchMethodEnum.RANDOM_SEARCH_SOBOL,
    #         outcome_constraints=["precision >= 0.7", "precision <= 1.0"],
    #         seed=5,
    #     )
    #     search_method.generate_evaluate_new_parameter_values(
    #         evaluate_recall_precision, arm_count=5
    #     )
    #     parameter_values_with_scores = search_method.list_parameter_value_scores()
    #     self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
    #     self.assertSetEqual(
    #         set(parameter_values_with_scores),
    #         {
    #             "trial_index",
    #             "arm_name",
    #             "parameters",
    #             "mean_recall",
    #             "sem_recall",
    #             "mean_precision",
    #             "sem_precision",
    #         },
    #     )
    #     self.assertEqual(
    #         (parameter_values_with_scores["mean_precision"] >= 0.7).sum(), 3
    #     )

    # def test_outcome_constraint_with_filter(self) -> None:
    #     def run_model(x):
    #         precision = x
    #         recall = -x + 1
    #         return recall, precision

    #     def evaluate_recall_precision(params: Dict[str, float]) -> Dict[str, Tuple]:
    #         recall, precision = run_model(params["x"])
    #         return {"recall": (recall, 0.0), "precision": (precision, 0.0)}

    #     search_method = tpt.SearchMethodFactory.create_search_method(
    #         parameters=[
    #             {
    #                 "name": "x",
    #                 "type": "range",
    #                 "value_type": "float",
    #                 "bounds": [0.0, 1.0],
    #             }
    #         ],
    #         selected_search_method=SearchMethodEnum.RANDOM_SEARCH_SOBOL,
    #         outcome_constraints=["precision >= 0.7", "precision <= 1.0"],
    #         seed=5,
    #     )
    #     search_method.generate_evaluate_new_parameter_values(
    #         evaluate_recall_precision, arm_count=5
    #     )
    #     parameter_values_with_scores = search_method.list_parameter_value_scores(
    #         legit_arms_only=True
    #     )
    #     self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
    #     self.assertSetEqual(
    #         set(parameter_values_with_scores),
    #         {
    #             "trial_index",
    #             "arm_name",
    #             "parameters",
    #             "mean_recall",
    #             "sem_recall",
    #             "mean_precision",
    #             "sem_precision",
    #         },
    #     )
    #     self.assertEqual(
    #         (parameter_values_with_scores["mean_precision"] >= 0.7).sum(), 3
    #     )
    #     self.assertEqual(
    #         (parameter_values_with_scores["mean_precision"] < 0.7).sum(), 0
    #     )


class TestSearchForMultipleSpaces(TestCase):
    def setUp(self):
        self.parameters = {
            "space1": [
                {
                    "name": "p",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "d",
                    "type": "choice",
                    "values": list(range(1, 3)),
                    "value_type": "int",
                    "is_ordered": True,
                },
                {
                    "name": "q",
                    "type": "choice",
                    "values": list(range(1, 6)),
                    "value_type": "int",
                    "is_ordered": True,
                },
            ],
            "space2": [
                {
                    "name": "n_changepoints",
                    "type": "choice",
                    "value_type": "int",
                    "values": list(range(15, 40, 5)),
                    "is_ordered": True,
                },
                {
                    "name": "yearly_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "weekly_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "daily_seasonality",
                    "type": "choice",
                    "value_type": "bool",
                    "values": [True, False],
                },
                {
                    "name": "seasonality_mode",
                    "type": "choice",
                    "value_type": "str",
                    "values": ["additive", "multiplicative"],
                },
            ],
        }

    def test_constructor(self) -> None:
        search_for_multiple_spaces = tpt.SearchForMultipleSpaces(
            parameters=self.parameters,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            experiment_name="experiment1",
            objective_name="objective1",
        )
        self.assertIsInstance(search_for_multiple_spaces.search_agent_dict, dict)
        self.assertEqual(len(search_for_multiple_spaces.search_agent_dict), 2)
        self.assertIsInstance(
            search_for_multiple_spaces.search_agent_dict["space1"], tpt.RandomSearch
        )
        self.assertIsInstance(
            search_for_multiple_spaces.search_agent_dict["space2"], tpt.RandomSearch
        )

    def test_generate_evaluate_new_parameter_values(self) -> None:
        search_for_multiple_spaces = tpt.SearchForMultipleSpaces(
            parameters=self.parameters,
            search_method=SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            experiment_name="experiment1",
            objective_name="objective1",
        )
        random_state = np.random.RandomState(seed=0)

        def arima_evaluation_function(params):
            error = random_state.random()
            sem = 0.0  # standard error of the mean of model's estimation error.
            return error, sem

        for _ in range(4):
            search_for_multiple_spaces.generate_evaluate_new_parameter_values(
                "space1", evaluation_function=arima_evaluation_function, arm_count=5
            )

        parameter_values_with_scores = (
            search_for_multiple_spaces.list_parameter_value_scores()
        )
        self.assertIsInstance(parameter_values_with_scores, dict)
        parameter_values_with_scores = (
            search_for_multiple_spaces.list_parameter_value_scores(
                selected_model="space1"
            )
        )
        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)
        parameter_values_with_scores = (
            search_for_multiple_spaces.list_parameter_value_scores(
                selected_model="space2"
            )
        )
        self.assertIsInstance(parameter_values_with_scores, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
