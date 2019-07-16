"""TBW."""
import logging
import numpy as np
import typing as t
try:
    import pygmo as pg
except ImportError:
    import warnings
    warnings.warn("Cannot import 'pygmo", RuntimeWarning, stacklevel=2)


from pyxel.calibration.fitting import ModelFitting
from pyxel.pipelines.model_function import ModelFunction
from pyxel.pipelines.processor import Processor
from pyxel.util import Outputs
import esapy_config.config as ec
from esapy_config import validators


# FRED: Put classes `Algorithm` and `Calibration` in separated files.

@ec.config
class Algorithm:
    """TBW.

    :return:
    """

    type = ec.setting(
        type=str,
        validator=validators.validate_in(['sade', 'sga', 'nlopt']),
        default='sade',
        doc=''
    )
    generations = ec.setting(
        type=int,
        validator=validators.interval(1, 100000),
        default=1,
        doc=''
    )
    population_size = ec.setting(
        type=int,
        validator=validators.interval(1, 100000),
        default=1,
        doc=''
    )

    # HANS: apply the coding conventions to the pyx.attribute below as they are vertically defined above.
    # FRED: Maybe a new class `Sade` could contains these attributes ?
    # SADE #####
    variant = ec.setting(type=int, validator=validators.interval(1, 18), default=2, doc='')
    variant_adptv = ec.setting(type=int, validator=validators.interval(1, 2), default=1, doc='')
    ftol = ec.setting(type=float, default=1e-06, doc='')  # validator=pyx.validate_range(),
    xtol = ec.setting(type=float, default=1e-06, doc='')  # validator=pyx.validate_range(),
    memory = ec.setting(type=bool, default=False, doc='')
    # SADE #####

    # FRED: Maybe a new class `SGA` could contains these attributes ?
    # SGA #####
    cr = ec.setting(type=float, converter=float, validator=validators.interval(0., 1.), default=0.9, doc='')
    eta_c = ec.setting(type=float, converter=float, default=1.0, doc='')
    m = ec.setting(type=float, converter=float, validator=validators.interval(0., 1.), default=0.02, doc='')
    param_m = ec.setting(type=float, default=1.0, doc='')            # validator=pyx.validate_range(1, 2),
    param_s = ec.setting(type=int, default=2, doc='')                # validator=pyx.validate_range(1, 2),
    crossover = ec.setting(type=str, default='exponential', doc='')  # validator=pyx.validate_choices(),
    mutation = ec.setting(type=str, default='polynomial', doc='')    # validator=pyx.validate_choices(),
    selection = ec.setting(type=str, default='tournament', doc='')   # validator=pyx.validate_choices(),
    # SGA #####

    # FRED: Maybe a new class `NLOPT` could contains these attributes ?
    # NLOPT #####
    nlopt_solver = ec.setting(type=str, default='neldermead', doc='')    # validator=pyx.validate_choices(),  todo
    maxtime = ec.setting(type=int, default=0, doc='')                     # validator=pyx.validate_range(),  todo
    maxeval = ec.setting(type=int, default=0, doc='')
    xtol_rel = ec.setting(type=float, default=1.e-8, doc='')
    xtol_abs = ec.setting(type=float, default=0., doc='')
    ftol_rel = ec.setting(type=float, default=0., doc='')
    ftol_abs = ec.setting(type=float, default=0., doc='')
    stopval = ec.setting(type=float, default=float('-inf'), doc='')
    local_optimizer = ec.setting(type=t.Optional[t.Any],
                                 default=None,
                                 doc='')          # validator=pyx.validate_choices(),  todo
    replacement = ec.setting(type=str, default='best', doc='')
    nlopt_selection = ec.setting(type=str, default='best', doc='')         # todo: "selection" - same name as in SGA
    # NLOPT #####

    # FRED: This could be refactored for each if-statement
    def get_algorithm(self) -> t.Any:
        """TBW.

        :return:
        """
        if self.type == 'sade':
            opt_algorithm = pg.sade(gen=self.generations,
                                    variant=self.variant,
                                    variant_adptv=self.variant_adptv,
                                    ftol=self.ftol,
                                    xtol=self.xtol,
                                    memory=self.memory)
        elif self.type == 'sga':
            opt_algorithm = pg.sga(gen=self.generations,
                                   cr=self.cr,                  # crossover probability
                                   crossover=self.crossover,    # single, exponential, binomial, sbx
                                   m=self.m,                    # mutation probability
                                   mutation=self.mutation,      # uniform, gaussian, polynomial
                                   param_s=self.param_s,        # number of best ind. in 'truncated'/tournament
                                   selection=self.selection,    # tournament, truncated
                                   eta_c=self.eta_c,            # distribution index for sbx crossover
                                   param_m=self.param_m)        # mutation parameter
        elif self.type == 'nlopt':
            opt_algorithm = pg.nlopt(self.nlopt_solver)
            opt_algorithm.maxtime = self.maxtime        # stop when the optimization time (in seconds) exceeds maxtime
            opt_algorithm.maxeval = self.maxeval        # stop when the number of function evaluations exceeds maxeval
            opt_algorithm.xtol_rel = self.xtol_rel      # relative stopping criterion for x
            opt_algorithm.xtol_abs = self.xtol_abs      # absolute stopping criterion for x
            opt_algorithm.ftol_rel = self.ftol_rel
            opt_algorithm.ftol_abs = self.ftol_abs
            opt_algorithm.stopval = self.stopval
            opt_algorithm.local_optimizer = self.local_optimizer
            opt_algorithm.replacement = self.replacement
            opt_algorithm.selection = self.nlopt_selection
        else:
            raise NotImplementedError

        return opt_algorithm


@ec.config
class Calibration:
    """TBW.

    :return:
    """

    calibration_mode = ec.setting(
        type=str,
        validator=validators.validate_in(['pipeline', 'single_model']),
        default='pipeline',
        doc=''
    )
    result_type = ec.setting(
        type=str,
        validator=validators.validate_in(['image', 'signal', 'pixel']),
        default='image',
        doc=''
    )
    result_fit_range = ec.setting(
        type=t.Optional[list],
        default=None,
        doc=''
    )
    result_input_arguments = ec.setting(
        type=t.Optional[list],
        default=None,
        doc=''
    )
    target_data_path = ec.setting(
        type=t.Optional[list],
        default=None,
        doc=''
    )
    target_fit_range = ec.setting(
        type=t.Optional[list],
        default=None,
        doc=''
    )
    fitness_function = ec.setting(
        type=t.Union[ModelFunction, str],
        default='',
        doc=''
    )
    algorithm = ec.setting(
        type=t.Union[Algorithm, str],
        default='',
        doc=''
    )
    parameters = ec.setting(
        type=t.Union[list, str],
        default='',
        doc=''
    )
    seed = ec.setting(
        type=int,
        validator=validators.interval(0, 100000),
        default=np.random.randint(0, 100000),
        doc=''
    )
    weighting_path = ec.setting(
        type=t.Optional[list],
        default=None,
        doc=''
    )

    def run_calibration(self, processor: Processor,
                        output: t.Optional[Outputs] = None) -> None:
        """TBW.

        :param processor: Processor object
        :param output: Output object
        :return:
        """
        pg.set_global_rng_seed(seed=self.seed)
        logging.info('Seed: %d', self.seed)
        output_files = (None, None)
        if output:
            output_files = output.create_files()

        fitting = ModelFitting(processor, self.parameters)

        settings = {
            'calibration_mode': self.calibration_mode,
            'generations': self.algorithm.generations,
            'population_size': self.algorithm.population_size,
            'simulation_output': self.result_type,
            'fitness_func': self.fitness_function,
            'target_output': self.target_data_path,
            'target_fit_range': self.target_fit_range,
            'out_fit_range': self.result_fit_range,
            'input_arguments': self.result_input_arguments,
            'weighting': self.weighting_path,
            'champions_file': output_files[0],
            'population_file': output_files[1]
        }
        # HANS: it may be better to pass this in as **settings. Need to discuss. There are many arguments.
        fitting.configure(settings)

        prob = pg.problem(fitting)
        opt_algorithm = self.algorithm.get_algorithm()
        algo = pg.algorithm(opt_algorithm)
        pop = pg.population(prob, size=self.algorithm.population_size)
        pop = algo.evolve(pop)

        champion_f = pop.champion_f
        champion_x = pop.champion_x
        return fitting.get_results(fitness=champion_f,
                                   parameter=champion_x)
