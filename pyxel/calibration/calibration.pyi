import typing as t

from ..pipelines.model_function import ModelFunction
from pyxel.pipelines.processor import Processor
from pyxel.util import Outputs

class Algorithm:
    def __init__(
        self,
        type: str = "sade",
        generations: int = 1,
        population_size: int = 1,
        variant: int = 2,
        variant_adptv: int = 1,
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        memory: bool = False,
        cr: float = 0.9,
        eta_c: float = 1.0,
        m: float = 0.02,
        param_m: float = 1.0,
        param_s: int = 2,
        crossover: str = "exponential",
        mutation: str = "polynomial",
        selection: str = "tournament",
        nlopt_solver: str = "neldermead",
        maxtime: int = 0,
        maxeval: int = 0,
        xtol_rel: float = 1.0e-8,
        xtol_abs: float = 0.0,
        ftol_rel: float = 0.0,
        ftol_abs: float = 0.0,
        stopval: float = float("-inf"),
        local_optimizer: t.Optional[t.Any] = None,
        replacement: str = "best",
        nlopt_selection: str = "best",
    ): ...
    type: str = ...
    generations: int = ...
    population_size: int = ...

    # SADE
    variant: int = ...
    variant_adptv: int = ...
    ftol: float = ...
    xtol: float = ...
    memory: bool = ...

    # SGA
    cr: float = ...
    eta_c: float = ...
    m: float = ...
    param_m: float = ...
    param_s: int = ...
    crossover: str = ...
    mutation: str = ...
    selection: str = ...

    # NLOPT
    nlopt_solver: str = ...
    maxtime: int = ...
    maxeval: int = ...
    xtol_rel: float = ...
    xtol_abs: float = ...
    ftol_rel: float = ...
    ftol_abs: float = ...
    stopval: float = ...
    local_optimizer: t.Optional[t.Any] = ...
    replacement: str = ...
    nlopt_selection: str = ...
    def get_algorithm(self) -> t.Any: ...

class Calibration:
    def __init__(
        self,
        calibration_mode: str = "pipeline",
        result_type: str = "image",
        result_fit_range: t.Optional[list] = None,
        result_input_arguments: t.Optional[list] = None,
        target_data_path: t.Optional[list] = None,
        target_fit_range: t.Optional[list] = None,
        fitness_function: t.Union[ModelFunction, str] = "",
        algorithm: t.Union[Algorithm, str] = "",
        parameters: t.Union[list, str] = "",
        seed: int = ...,
        weighting_path: t.Optional[list] = None,
    ): ...
    calibration_mode: str = ...
    result_type: str = ...
    result_fit_range: t.Optional[list] = ...
    result_input_arguments: t.Optional[list] = ...
    target_data_path: t.Optional[list] = ...
    target_fit_range: t.Optional[list] = ...

    fitness_function: t.Union[ModelFunction, str] = ...
    algorithm: t.Union[Algorithm, str] = ...
    parameters: t.Union[list, str] = ...
    seed: int = ...
    weighting_path: t.Optional[list] = ...
    def run_calibration(
        self, processor: Processor, output: t.Optional[Outputs] = None
    ) -> None: ...
