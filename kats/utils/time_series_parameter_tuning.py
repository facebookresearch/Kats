# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Module that has parameter tuning classes for time series models.

This module has a collection of classes. A subset of these classes are parameter tuning
strategies with their abstract parent class. In addition, there are helper classes,
such as a factory that creates search strategy objects.

  Typical usage example:

  >>> import time_series_parameter_tuning as tspt
  >>> a_search_strategy = tspt.SearchMethodFactory.create_search_method(...)
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from functools import reduce
from multiprocessing.pool import Pool
from numbers import Number
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from ax import Arm, ComparisonOp, Data, OptimizationConfig, SearchSpace
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.outcome_constraint import OutcomeConstraint
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.instantiation import (
    outcome_constraint_from_str,
    parameter_from_json,
)
from kats.consts import SearchMethodEnum

# Maximum number of worker processes used to evaluate trial arms in parallel
MAX_NUM_PROCESSES = 50


class Final(type):
    """A helper class to ensure a class cannot be inherited.

    It is used as:
        class Foo(metaclass=Final):
            ...

    Once the class, Foo, is declared in this way, no other class can
    inherit it. See the declaration of SearchMethodFactory class below.

    Attributes:
        N/A
    """

    def __new__(metacls, name, bases, classdict):
        """Checks if child class is instantiated. Throws an error if so.

        Args:
            metacls: To be used by metaclass argument of a new class instantiation
            name: Same as above
            bases: Same as above
            classdict: Same as above

        Returns:
            Type of the new class

        Raises:
            TypeError:
                Raised when an object of a class using this Final
                class as metaclass is created.
        """

        for b in bases:
            if isinstance(b, Final):
                raise TypeError(
                    "type '{0}' is not an acceptable base type".format(b.__name__)
                )
        return type.__new__(metacls, name, bases, dict(classdict))


class TimeSeriesEvaluationMetric(Metric):
    """Object to evaluate an arm

    An object of this class is used to evaluate an arm through search. It is mainly
    used to parallelize the search, as evaluation of an arm needs to be run in
    parallel. Obviously, this is possible if the search strategy allows it in
    theory.

    Attributes:
        evaluation_function: The name of the function to be used in evaluation.
        logger: the logger object to log.
        multiprocessing: Flag to decide whether evaluation will run in parallel.


    """

    def __init__(
        self,
        name: str,
        evaluation_function: Callable,
        logger: logging.Logger,
        multiprocessing: bool = False,
    ) -> None:
        super().__init__(name)
        self.evaluation_function = evaluation_function
        self.logger = logger
        self.multiprocessing = multiprocessing

    @classmethod
    def is_available_while_running(cls) -> bool:
        """Metrics are available while the trial is `RUNNING` and should
        always be re-fetched.
        """

        return True

    def evaluate_arm(self, arm) -> Dict:
        """Evaluates the performance of an arm.

        Takes an arm object, gets its parameter values, runs
        evaluation_function and returns what that function returns
        after reformatting it.

        Args:
            arm: The arm object to be evaluated.

        Returns:
            Either a dict or a list of dict. These dict objects need
            to have metric name that describes the metric, arm_name,
            mean which is the mean of the evaluation value and its
            standard error.
        """

        # Arm evaluation requires mean and standard error or dict for multiple metrics
        evaluation_result = self.evaluation_function(arm.parameters)
        if isinstance(evaluation_result, dict):
            return [
                {
                    "metric_name": name,
                    "arm_name": arm.name,
                    "mean": value[0],
                    "sem": value[1],
                }
                for (name, value) in evaluation_result.items()
            ]
        elif isinstance(evaluation_result, Number):
            evaluation_result = (evaluation_result, 0.0)
        elif (
            isinstance(evaluation_result, tuple)
            and len(evaluation_result) == 2
            and all(isinstance(n, Number) for n in evaluation_result)
        ):
            pass
        else:
            raise TypeError(
                "Evaluation function should either return a single numeric "
                "value that represents the error or a tuple of two numeric "
                "values, one for the mean of error and the other for the "
                "standard error of the mean of the error."
            )
        return {
            "metric_name": self.name,
            "arm_name": arm.name,
            "mean": evaluation_result[0],
            "sem": evaluation_result[1],
        }

    # pyre-fixme[14]: `fetch_trial_data` overrides method defined in `Metric`
    #  inconsistently.
    # pyre-fixme[14]: `fetch_trial_data` overrides method defined in `Metric`
    #  inconsistently.
    def fetch_trial_data(self, trial) -> Data:
        """Calls evaluation of every arm in a trial.

        Args:
            trial: The trial of which all arms to be evaluated.

        Returns:
            Data object that has arm names, trial index, evaluation.
        """

        if self.multiprocessing:
            with Pool(processes=min(len(trial.arms), MAX_NUM_PROCESSES)) as pool:
                records = pool.map(self.evaluate_arm, trial.arms)
                pool.close()
        else:
            records = list(map(self.evaluate_arm, trial.arms))
        if isinstance(records[0], list):
            # Evaluation result output contains multiple metrics
            records = [metric for record in records for metric in record]
        for record in records:
            record.update({"trial_index": trial.index})
        return Data(df=pd.DataFrame.from_records(records))


class TimeSeriesParameterTuning(ABC):
    """Abstract class for search strategy class, such as GridSearch, RandomSearch.

    Defines and imposes a structure to search strategy classes. Each search
    strategy has to have attributes listed below. Also, it provides methods
    that are common to search strategies.

    Attributes:
        parameters: List of dictionaries where each dict represents a hyperparameter.
        experiment_name: An arbitrary name for the experiment object.
        objective_name: An arbitrary name for the objective function that is used
            in the evaluation function.
        outcome_constraints: Constraints set on the outcome of the objective.
    """

    evaluation_function: Optional[Callable] = None
    outcome_constraints: Optional[List[OutcomeConstraint]] = None

    def __init__(
        self,
        parameters: Optional[List[Dict]] = None,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        outcome_constraints: Optional[List[str]] = None,
        multiprocessing: bool = False,
    ) -> None:
        if parameters is None:
            parameters = [{}]
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Parameter tuning search space dimensions: {}".format(parameters)
        )
        self.validate_parameters_format(parameters)
        self.parameters = [parameter_from_json(parameter) for parameter in parameters]
        self.outcome_constraints = (
            [
                outcome_constraint_from_str(str_constraint)
                for str_constraint in outcome_constraints
            ]
            if outcome_constraints is not None
            else None
        )
        self._kats_search_space = SearchSpace(parameters=self.parameters)
        self.logger.info("Search space is created.")
        self.job_id = uuid.uuid4()
        self.experiment_name = (
            experiment_name if experiment_name else f"parameter_tuning_{self.job_id}"
        )
        self.objective_name = (
            objective_name if objective_name else f"objective_{self.job_id}"
        )
        self.multiprocessing = multiprocessing

        self._exp = Experiment(
            name=self.experiment_name,
            search_space=self._kats_search_space,
            runner=SyntheticRunner(),
        )
        self._trial_data = Data()
        self.logger.info("Experiment is created.")

    @staticmethod
    def validate_parameters_format(parameters: List) -> None:
        """Check parameters objects structure.

        parameters object needs to be in a specific format. It needs to be
        a list of dict where each dict associates a parameter. Raises an
        error depending on the format violation.

        Args:
            parameters: parameters of which format is to be audited.

        Returns:
            None, if none of the checks fail, raises error if any fails.

        Raises:
            TypeError: If parameters is not of type list.
            ValueError: Parameters cannot be empty as there should be at least
                one hyperparameter to tune.
            TypeError: If any of the list element is of type other then dict
        """

        if not isinstance(parameters, list):
            raise TypeError(
                "The input parameter, parameters, should be a list even if a "
                "single parameter is defined."
            )
        if len(parameters) == 0:
            raise ValueError(
                "The parameter list is empty. No search space can be created "
                "if not parameter is specified."
            )
        for i, parameter_dict in enumerate(parameters):
            if not isinstance(parameter_dict, dict):
                raise TypeError(
                    "The parameter_dict, {i}, in the list of parameters should"
                    " be a dict. The parameter_dict is {parameter_dict}, and"
                    " its type is {type_}.".format(
                        i=i,
                        parameter_dict=str(parameter_dict),
                        type_=type(parameter_dict),
                    )
                )
            if len(parameter_dict) == 0:
                raise ValueError(
                    "A parameter_dict in the parameter list is empty. All "
                    "parameter_dict items should have valid key: value entries"
                    "."
                )

    def get_search_space(self):
        """Getter of search space attribute of the private attribute, _exp."""

        return self._exp.search_space

    def generator_run_for_search_method(
        self, evaluation_function: Callable, generator_run: DiscreteModelBridge
    ) -> None:
        """Creates a new batch trial then runs the lastest.

        Args:
            evaluation_function: The name of the function to use for arm evaluation
            generator_run: Generator_run object that is used to populate new arms
        """

        self.evaluation_function = evaluation_function
        outcome_constraints = self.outcome_constraints
        if outcome_constraints:
            # Convert dummy base Metrics to TimeseriesEvaluationMetrics
            self.outcome_constraints = [
                OutcomeConstraint(
                    TimeSeriesEvaluationMetric(
                        name=oc.metric.name,
                        evaluation_function=evaluation_function,
                        logger=self.logger,
                        multiprocessing=self.multiprocessing,
                    ),
                    op=oc.op,
                    bound=oc.bound,
                    relative=oc.relative,
                )
                for oc in outcome_constraints
            ]
        self._exp.optimization_config = OptimizationConfig(
            objective=Objective(
                metric=TimeSeriesEvaluationMetric(
                    name=self.objective_name,
                    evaluation_function=self.evaluation_function,
                    logger=self.logger,
                    multiprocessing=self.multiprocessing,
                ),
                minimize=True,
            ),
            outcome_constraints=self.outcome_constraints,
        )

        # pyre-fixme[6]: Expected `Optional[GeneratorRun]` for 1st param but got
        #  `DiscreteModelBridge`.
        self._exp.new_batch_trial(generator_run=generator_run)
        # We run the most recent batch trial as we only run candidate trials
        self._exp.trials[max(self._exp.trials)].run()
        self._trial_data = Data.from_multiple_data(
            # pyre-fixme[6]: Expected `Iterable[ax.core.data.Data]` for 1st param
            #  but got `Iterable[ax.core.abstract_data.AbstractDataFrameData]`.
            [
                self._trial_data,
                self._exp.fetch_trials_data(trial_indices=[max(self._exp.trials)]),
            ]
        )

    @abstractmethod
    def generate_evaluate_new_parameter_values(
        self,
        evaluation_function: Callable,
        arm_count: int = -1  # -1 means
        # create all arms (i.e. all combinations of parameter values)
    ) -> None:
        """A place holder method for users that are still using it.

        It previously ran evaluation for trials. That part was moved to
        generator_run_for_search_methods(). Now this method does nothing.
        """

        pass

    @staticmethod
    def _repivot_dataframe(armscore_df: pd.DataFrame):
        """Reformats the score data frame.

        Args:
            armscore_df: Pandas DataFrame object that has the arm scores
                in raw format.

        Returns:
            Pandas DataFrame object of arm score in the new format
        """

        transform = (
            armscore_df.set_index(["trial_index", "arm_name", "metric_name"])
            .unstack("metric_name")
            .reset_index()
        )
        new_cols = transform.columns.to_flat_index()
        parameters_holder = transform[
            list(filter(lambda x: "parameters" in x, new_cols))[0]
        ]
        transform.drop(columns="parameters", level=0, inplace=True)
        new_cols = new_cols.drop(labels=filter(lambda x: "parameters" in x, new_cols))
        transform.columns = ["trial_index", "arm_name"] + [
            "_".join(tpl) for tpl in new_cols[2:]
        ]
        transform["parameters"] = parameters_holder
        return transform

    def list_parameter_value_scores(
        self, legit_arms_only: bool = False
    ) -> pd.DataFrame:
        """Creates a Pandas DataFrame from evaluated arms then returns it.

        The method should be called to fetch evaluation results of arms that
        are populated and evaluated so far.

        Args:
            legit_arms_only: A flag to filter arms that violate output_constraints
                if given any.

        Returns:
            A Pandas DataFrame that holds arms populated and evaluated so far.
        """

        # For experiments which have not ran generate_evaluate_new_parameter_values,
        # we cannot provide trial data without metrics, so we return empty dataframe
        if not self._exp.metrics:
            return pd.DataFrame(
                [],
                columns=[
                    "arm_name",
                    "metric_name",
                    "mean",
                    "sem",
                    "parameters",
                    "trial_index",
                ],
            )
        armscore_df = self._trial_data.df.copy()
        armscore_df["parameters"] = armscore_df["arm_name"].map(
            {k: v.parameters for k, v in self._exp.arms_by_name.items()}
        )
        if self.outcome_constraints:
            # Deduplicate entries for which there are outcome constraints
            armscore_df = armscore_df.loc[
                # pyre-ignore[16]: `None` has no attribute `index`.
                armscore_df.astype(str)
                .drop_duplicates()
                .index
            ]
            if legit_arms_only:

                def filter_violating_arms(
                    arms: List[Arm], data: Data, optimization_config: OptimizationConfig
                ) -> List[Arm]:
                    boolean_indices = []
                    for oc in optimization_config.outcome_constraints:
                        if oc.op is ComparisonOp.LEQ:
                            boolean_indices.append(
                                data.df[data.df.metric_name == oc.metric.name]["mean"]
                                <= oc.bound
                            )
                        else:
                            boolean_indices.append(
                                data.df[data.df.metric_name == oc.metric.name]["mean"]
                                >= oc.bound
                            )
                    eligible_arm_indices = reduce(lambda x, y: x & y, boolean_indices)
                    eligible_arm_names = data.df.loc[eligible_arm_indices.index][
                        eligible_arm_indices
                    ].arm_name
                    return list(
                        filter(lambda x: x.name in eligible_arm_names.values, arms)
                    )

                filtered_arms = filter_violating_arms(
                    list(self._exp.arms_by_name.values()),
                    # pyre-fixme[6]: Expected `Data` for 2nd param but got
                    #  `AbstractDataFrameData`.
                    self._exp.fetch_data(),
                    # pyre-fixme[6]: Expected `OptimizationConfig` for 3rd param but
                    #  got `Optional[ax.core.optimization_config.OptimizationConfig]`.
                    self._exp.optimization_config,
                )
                armscore_df = armscore_df[
                    armscore_df["arm_name"].isin([arm.name for arm in filtered_arms])
                ]
            armscore_df = self._repivot_dataframe(armscore_df)
        return armscore_df


class SearchMethodFactory(metaclass=Final):
    """Generates and returns  search strategy object."""

    def __init__(self):
        raise TypeError(
            "SearchMethodFactory is not allowed to be instantiated. Use "
            "it as a static class."
        )

    @staticmethod
    def create_search_method(
        parameters: List[Dict],
        selected_search_method: SearchMethodEnum = SearchMethodEnum.GRID_SEARCH,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        outcome_constraints: Optional[List[str]] = None,
        seed: Optional[int] = None,
        bootstrap_size: int = 5,
        evaluation_function: Optional[Callable] = None,
        bootstrap_arms_for_bayes_opt: Optional[List[dict]] = None,
        multiprocessing: bool = False,
    ) -> TimeSeriesParameterTuning:
        """The static method of factory class that creates the search method
        object. It does not require the class to be instantiated.

        Args:
            parameters: List[Dict] = None,
                Defines parameters by their names, their types their optional
                values for custom parameter search space.
            selected_search_method: SearchMethodEnum = SearchMethodEnum.GRID_SEARCH
                Defines search method to be used during parameter tuning. It has to
                be an option from the enum, SearchMethodEnum.
            experiment_name: str = None,
                Name of the experiment to be used in Ax's experiment object.
            objective_name: str = None,
                Name of the objective to be used in Ax's experiment evaluation.
            outcome_constraints: List[str] = None
                List of constraints defined as strings. Example: ['metric1 >= 0',
                'metric2 < 5]
            bootstrap_arms_for_bayes_opt: List[dict] = None
                List of params. It provides a list of self-defined inital parameter
                values for Baysian Optimal search. Example: for Holt Winter's model,
                [{'m': 7}, {'m': 14}]

        Returns:
            A search object, GridSearch, RandomSearch, or BayesianOptSearch,
                depending on the selection.

        Raises:
            NotImplementedError: Raised if the selection is not among strategies
                that are implemented.
        """

        if selected_search_method == SearchMethodEnum.GRID_SEARCH:
            return GridSearch(
                parameters=parameters,
                experiment_name=experiment_name,
                objective_name=objective_name,
                outcome_constraints=outcome_constraints,
                multiprocessing=multiprocessing,
            )
        elif (
            selected_search_method == SearchMethodEnum.RANDOM_SEARCH_UNIFORM
            or selected_search_method == SearchMethodEnum.RANDOM_SEARCH_SOBOL
        ):
            return RandomSearch(
                parameters=parameters,
                experiment_name=experiment_name,
                objective_name=objective_name,
                random_strategy=selected_search_method,
                outcome_constraints=outcome_constraints,
                seed=seed,
                multiprocessing=multiprocessing,
            )
        elif selected_search_method == SearchMethodEnum.BAYES_OPT:
            assert (
                evaluation_function is not None
            ), "evaluation_function cannot be None. It is needed at initialization of BayesianOptSearch object."
            return BayesianOptSearch(
                parameters=parameters,
                evaluation_function=evaluation_function,
                experiment_name=experiment_name,
                objective_name=objective_name,
                bootstrap_size=bootstrap_size,
                seed=seed,
                bootstrap_arms_for_bayes_opt=bootstrap_arms_for_bayes_opt,
                outcome_constraints=outcome_constraints,
                multiprocessing=multiprocessing,
            )
        else:
            raise NotImplementedError(
                "A search method yet to implement is selected. Only grid"
                " search and random search are implemented."
            )


class GridSearch(TimeSeriesParameterTuning):
    """The method factory class that creates the search method object. It does
    not require the class to be instantiated.

    Do not instantiate this class using its constructor.
    Rather use the factory, SearchMethodFactory.

    Attributes:
        parameters: List[Dict] = None,
            Defines parameters by their names, their types their optional
            values for custom parameter search space.
        experiment_name: str = None,
            Name of the experiment to be used in Ax's experiment object.
        objective_name: str = None,
            Name of the objective to be used in Ax's experiment evaluation.
        outcome_constraints: List[str] = None
            List of constraints defined as strings. Example: ['metric1 >= 0',
            'metric2 < 5]
    """

    def __init__(
        self,
        parameters: List[Dict],
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        outcome_constraints: Optional[List[str]] = None,
        multiprocessing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            parameters,
            experiment_name,
            objective_name,
            outcome_constraints,
            multiprocessing,
        )
        self._factorial = Models.FACTORIAL(
            search_space=self.get_search_space(), check_cardinality=False
        )
        self.logger.info("A factorial model for arm generation is created.")
        self.logger.info("A GridSearch object is successfully created.")

    def generate_evaluate_new_parameter_values(
        self,
        evaluation_function: Callable,
        arm_count: int = -1,  # -1 means create all arms (i.e. all combinations of
        # parameter values)
    ) -> None:
        """This method can only be called once. arm_count other than -1 will be ignored
        as this search strategy exhaustively explores all arms.
        """

        if arm_count != -1:
            # FullFactorialGenerator ignores specified arm_count as it automatically determines how many arms
            self.logger.info(
                "GridSearch arm_count input is ignored and automatically determined by generator."
            )
            arm_count = -1
        factorial_run = self._factorial.gen(n=arm_count)
        self.generator_run_for_search_method(
            evaluation_function=evaluation_function, generator_run=factorial_run
        )


class RandomSearch(TimeSeriesParameterTuning):
    """Random search for hyperparameter tuning.

    Do not instantiate this class using its constructor.
    Rather use the factory, SearchMethodFactory.

    Attributes:
        parameters: List[Dict],
            Defines parameters by their names, their types their optional
            values for custom parameter search space.
        experiment_name: str = None,
            Name of the experiment to be used in Ax's experiment object.
        objective_name: str = None,
            Name of the objective to be used in Ax's experiment evaluation.
        seed: int = None,
            Seed for Ax quasi-random model. If None, then time.time() is set.
        random_strategy: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            By now, we already know that the search method is random search.
            However, there are optional random strategies: UNIFORM, or SOBOL.
            This parameter allows to select it.
        outcome_constraints: List[str] = None
            List of constraints defined as strings. Example: ['metric1 >= 0',
            'metric2 < 5]
    """

    def __init__(
        self,
        parameters: List[Dict],
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        seed: Optional[int] = None,
        random_strategy: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        outcome_constraints: Optional[List[str]] = None,
        multiprocessing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            parameters,
            experiment_name,
            objective_name,
            outcome_constraints,
            multiprocessing,
        )
        if seed is None:
            seed = int(time.time())
            self.logger.info(
                "No seed is given by the user, it will be set by the current time"
            )
        self.logger.info("Seed that is used in random search: {seed}".format(seed=seed))
        if random_strategy == SearchMethodEnum.RANDOM_SEARCH_UNIFORM:
            self._random_strategy_model = Models.UNIFORM(
                search_space=self.get_search_space(), deduplicate=True, seed=seed
            )
        elif random_strategy == SearchMethodEnum.RANDOM_SEARCH_SOBOL:
            self._random_strategy_model = Models.SOBOL(
                search_space=self.get_search_space(), deduplicate=True, seed=seed
            )
        else:
            raise NotImplementedError(
                "Invalid random strategy selection. It should be either "
                "uniform or sobol."
            )
        self.logger.info(
            "A {random_strategy} model for candidate parameter value generation"
            " is created.".format(random_strategy=random_strategy)
        )
        self.logger.info("A RandomSearch object is successfully created.")

    def generate_evaluate_new_parameter_values(
        self, evaluation_function: Callable, arm_count: int = 1
    ) -> None:
        """This method can be called as many times as desired with arm_count in
        desired number. The total number of generated candidates will be equal
        to the their multiplication. Suppose we would like to sample k
        candidates where k = m x n such that k, m, n are integers. We can call
        this function once with `arm_count=k`, or call it k time with
        `arm_count=1` (or without that parameter at all), or call it n times
        `arm_count=m` and vice versa. They all will yield k candidates, however
        it is not guaranteed that the candidates will be identical across these
        scenarios.
        """

        model_run = self._random_strategy_model.gen(n=arm_count)
        self.generator_run_for_search_method(
            evaluation_function=evaluation_function, generator_run=model_run
        )


class BayesianOptSearch(TimeSeriesParameterTuning):
    """Bayesian optimization search for hyperparameter tuning.

    Do not instantiate this class using its constructor.
    Rather use the factory, SearchMethodFactory.

    Attributes:
        parameters: List[Dict],
            Defines parameters by their names, their types their optional
            values for custom parameter search space.
        evaluation_function: Callable
            The evaluation function to pass to Ax to evaluate arms.
        experiment_name: str = None,
            Name of the experiment to be used in Ax's experiment object.
        objective_name: str = None,
            Name of the objective to be used in Ax's experiment evaluation.
        bootstrap_size: int = 5,
            The number of arms that will be randomly generated to bootstrap the
            Bayesian optimization.
        seed: int = None,
            Seed for Ax quasi-random model. If None, then time.time() is set.
        random_strategy: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
            By now, we already know that the search method is random search.
            However, there are optional random strategies: UNIFORM, or SOBOL.
            This parameter allows to select it.
        outcome_constraints: List[str] = None
            List of constraints defined as strings. Example: ['metric1 >= 0',
            'metric2 < 5]
    """

    # pyre-fixme[11]: Annotation `BOTORCH` is not defined as a type.
    _bayes_opt_model: Optional[Models.BOTORCH] = None

    def __init__(
        self,
        parameters: List[Dict],
        evaluation_function: Callable,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        bootstrap_size: int = 5,
        seed: Optional[int] = None,
        random_strategy: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        outcome_constraints: Optional[List[str]] = None,
        multiprocessing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            parameters,
            experiment_name,
            objective_name,
            outcome_constraints,
            multiprocessing,
        )
        if seed is None:
            seed = int(time.time())
            self.logger.info(
                "No seed is given by the user, it will be set by the current time"
            )
        self.logger.info("Seed that is used in random search: {seed}".format(seed=seed))
        if random_strategy == SearchMethodEnum.RANDOM_SEARCH_UNIFORM:
            self._random_strategy_model = Models.UNIFORM(
                search_space=self.get_search_space(), deduplicate=True, seed=seed
            )
        elif random_strategy == SearchMethodEnum.RANDOM_SEARCH_SOBOL:
            self._random_strategy_model = Models.SOBOL(
                search_space=self.get_search_space(), deduplicate=True, seed=seed
            )
        else:
            raise NotImplementedError(
                "Invalid random strategy selection. It should be either "
                "uniform or sobol."
            )
        self.logger.info(
            "A {random_strategy} model for candidate parameter value generation"
            " is created.".format(random_strategy=random_strategy)
        )

        bootstrap_arms_for_bayes_opt = kwargs.get("bootstrap_arms_for_bayes_opt", None)
        if bootstrap_arms_for_bayes_opt is None:
            model_run = self._random_strategy_model.gen(n=bootstrap_size)
        else:
            bootstrap_arms_list = [
                Arm(name="0_" + str(i), parameters=params)
                for i, params in enumerate(bootstrap_arms_for_bayes_opt)
            ]
            model_run = GeneratorRun(bootstrap_arms_list)

        self.generator_run_for_search_method(
            evaluation_function=evaluation_function, generator_run=model_run
        )
        self.logger.info(f'fitted data columns: {self._trial_data.df["metric_name"]}')
        self.logger.info(f"Bootstrapping of size = {bootstrap_size} is done.")

    def generate_evaluate_new_parameter_values(
        self, evaluation_function: Callable, arm_count: int = 1
    ) -> None:
        """This method can be called as many times as desired with arm_count in
        desired number. The total number of generated candidates will be equal
        to the their multiplication. Suppose we would like to sample k
        candidates where k = m x n such that k, m, n are integers. We can call
        this function once with `arm_count=k`, or call it k time with
        `arm_count=1` (or without that parameter at all), or call it n times
        `arm_count=m` and vice versa. They all will yield k candidates, however
        it is not guaranteed that the candidates will be identical across these
        scenarios. We re-initiate BOTORCH model on each call.
        """

        self._bayes_opt_model = Models.BOTORCH(
            experiment=self._exp,
            data=self._trial_data,
        )
        model_run = self._bayes_opt_model.gen(n=arm_count)
        self.generator_run_for_search_method(
            evaluation_function=evaluation_function,
            # pyre-fixme[6]: Expected `DiscreteModelBridge` for 2nd param but got
            #  `GeneratorRun`.
            generator_run=model_run,
        )


class SearchForMultipleSpaces:
    def __init__(
        self,
        parameters: Dict[str, List[Dict]],
        search_method: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Search class that runs search for multiple search spaces.

        Created and used for ensemble models, or model selection.

        Attributes:
            parameters: Dict[str, List[Dict]],
                Defines a search space per model. It maps model names to search spaces
            experiment_name: str = None,
                Name of the experiment to be used in Ax's experiment object.
            objective_name: str = None,
                Name of the objective to be used in Ax's experiment evaluation.
            seed: int = None,
                Seed for Ax quasi-random model. If None, then time.time() is set.
            random_strategy: SearchMethodEnum = SearchMethodEnum.RANDOM_SEARCH_UNIFORM,
                By now, we already know that the search method is random search.
                However, there are optional random strategies: UNIFORM, or SOBOL.
                This parameter allows to select it.
        """

        # search_agent_dict is a dict for str -> TimeSeriesParameterTuning object
        # Thus, we can access different search method objects created using their
        # keys.
        self.search_agent_dict = {
            agent_name: SearchMethodFactory.create_search_method(
                parameters=model_params,
                selected_search_method=search_method,
                experiment_name=experiment_name,
                objective_name=objective_name,
                seed=seed,
            )
            for agent_name, model_params in parameters.items()
        }

    def generate_evaluate_new_parameter_values(
        self, selected_model: str, evaluation_function: Callable, arm_count: int = 1
    ) -> None:
        """Calls generate_evaluate_new_parameter_values() for the search method in
        the search methods collection, search_agent_dict, called by selection_model
        name.

        Args:
            selected_model: The name of the model that is being tuned for.
                evaluation_function: The evaluation function to be used to evaluate
                arms.
            arm_count: Number of arms to be popuelated and evaluated.
        """

        self.search_agent_dict[selected_model].generate_evaluate_new_parameter_values(
            evaluation_function=evaluation_function, arm_count=arm_count
        )

    def list_parameter_value_scores(
        self, selected_model: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Calls list_parameter_value_scores() for the model that the name is given
        or calls for every model otherwise.

        Args:
            select_model: The name of the model of which the agent's
                list_parameter_value_scores() will be called, if given. If None,
                then the same method is called for all model.

        Returns:
            A dictionary in which keys are model names, values are associated score
            data frames.
        """

        if selected_model:
            return self.search_agent_dict[selected_model].list_parameter_value_scores()
        else:  # selected_model is not provided, therefore this method will
            # return a dict of data frames where each key points to the
            # parameter score values of the corresponding models.
            return {
                selected_model_: self.search_agent_dict[
                    selected_model_
                ].list_parameter_value_scores()
                for selected_model_ in self.search_agent_dict
            }
