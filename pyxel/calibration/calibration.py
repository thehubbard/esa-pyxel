"""TBW."""
import logging
import numpy as np
import pygmo as pg
import pyxel as pyx
from pyxel.calibration.fitting import ModelFitting
from pyxel.pipelines.model_function import ModelFunction
from pyxel.pipelines.processor import Processor


@pyx.detector_class
class Algorithm:
    """TBW.

    :return:
    """

    type = pyx.attribute(
        type=str,
        validator=[pyx.validate_type(str),
                   pyx.validate_choices(['sade', 'sga', 'nlopt'])],
        default='sade',
        doc=''
    )
    generations = pyx.attribute(
        type=int,
        validator=[pyx.validate_type(int),
                   pyx.validate_range(1, 100000)],
        default=1,
        doc=''
    )
    population_size = pyx.attribute(
        type=int,
        validator=[pyx.validate_type(int),
                   pyx.validate_range(1, 100000)],
        default=1,
        doc=''
    )

    # SADE #####
    variant = pyx.attribute(type=int, validator=[pyx.validate_type(int),
                                                 pyx.validate_range(1, 18)], default=2, doc='')
    variant_adptv = pyx.attribute(type=int, validator=[pyx.validate_type(int),
                                                       pyx.validate_range(1, 2)], default=1, doc='')
    ftol = pyx.attribute(type=float, default=1e-06, doc='')  # validator=pyx.validate_range(),
    xtol = pyx.attribute(type=float, default=1e-06, doc='')  # validator=pyx.validate_range(),
    memory = pyx.attribute(type=bool, default=False, doc='')
    # SADE #####

    # SGA #####
    cr = pyx.attribute(type=float, converter=float, validator=[pyx.validate_type(float),
                                                               pyx.validate_range(0, 1)], default=0.9, doc='')
    eta_c = pyx.attribute(type=float, converter=float, validator=[pyx.validate_type(float)], default=1.0, doc='')
    m = pyx.attribute(type=float, converter=float, validator=[pyx.validate_type(float),
                                                              pyx.validate_range(0, 1)], default=0.02, doc='')
    param_m = pyx.attribute(type=float, default=1.0, doc='')   # validator=pyx.validate_range(1, 2),
    param_s = pyx.attribute(type=int, default=2, doc='')  # validator=pyx.validate_range(1, 2),
    crossover = pyx.attribute(type=str, default='exponential', doc='')  # validator=pyx.validate_choices(),
    mutation = pyx.attribute(type=str, default='polynomial', doc='')  # validator=pyx.validate_choices(),
    selection = pyx.attribute(type=str, default='tournament', doc='')   # validator=pyx.validate_choices(),
    # SGA #####

    # NLOPT #####
    nlopt_solver = pyx.attribute(type=str, default='neldermead', doc='')    # validator=pyx.validate_choices(),  todo
    maxtime = pyx.attribute(type=int, default=0, doc='')                     # validator=pyx.validate_range(),  todo
    maxeval = pyx.attribute(type=int, default=0, doc='')
    xtol_rel = pyx.attribute(type=float, default=1.e-8, doc='')
    xtol_abs = pyx.attribute(type=float, default=0., doc='')
    ftol_rel = pyx.attribute(type=float, default=0., doc='')
    ftol_abs = pyx.attribute(type=float, default=0., doc='')
    stopval = pyx.attribute(type=float, default=float('-inf'), doc='')
    local_optimizer = pyx.attribute(type=None, default=None, doc='')          # validator=pyx.validate_choices(),  todo
    replacement = pyx.attribute(type=str, default='best', doc='')
    nlopt_selection = pyx.attribute(type=str, default='best', doc='')         # todo: "selection" - same name as in SGA
    # NLOPT #####

    def get_algorithm(self):
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


@pyx.detector_class
class Calibration:
    """TBW.

    :return:
    """

    calibration_mode = pyx.attribute(
        type=str,
        validator=[pyx.validate_type(str),
                   pyx.validate_choices(['pipeline', 'single_model'])],
        default='pipeline',
        doc=''
    )
    output_type = pyx.attribute(
        type=str,
        validator=[pyx.validate_type(str),
                   pyx.validate_choices(['image', 'signal', 'pixel'])],
        default='image',
        doc=''
    )
    output_fit_range = pyx.attribute(
        type=list,
        validator=[pyx.validate_type(list)],
        default=None,
        doc=''
    )
    target_data_path = pyx.attribute(
        type=list,
        validator=[pyx.validate_type(list)],
        default=None,
        doc=''
    )
    target_fit_range = pyx.attribute(
        type=list,
        validator=[pyx.validate_type(list)],
        default=None,
        doc=''
    )
    fitness_function = pyx.attribute(
        type=ModelFunction,
        validator=[pyx.validate_type(ModelFunction)],
        default='',
        doc=''
    )
    algorithm = pyx.attribute(
        type=Algorithm,
        validator=[pyx.validate_type(Algorithm)],
        default='',
        doc=''
    )
    parameters = pyx.attribute(
        type=list,
        validator=[pyx.validate_type(list)],
        default='',
        doc=''
    )
    seed = pyx.attribute(
        type=int,
        validator=[pyx.validate_type(int),
                   pyx.validate_range(0, 100000)],
        default=np.random.randint(0, 100000),
        doc=''
    )
    weighting_path = pyx.attribute(
        type=list,
        # validator=[pyx.validate_type(list)],  # todo
        default=None,
        doc=''
    )
    champions_file = pyx.attribute(          # TODO: move to output
        type=str,
        # validator=[pyx.validate_type(str)],
        default='data/calibration_champions.out',
        doc=''
    )
    population_file = pyx.attribute(          # TODO: move to output
        type=str,
        # validator=[pyx.validate_type(str)],
        default=None,
        doc=''
    )

    def run_calibration(self, processor: Processor,
                        output_files: tuple):
        """TBW.

        :param processor:
        :param output_files: tuple
        :return:
        """
        pg.set_global_rng_seed(seed=self.seed)
        logger = logging.getLogger('pyxel')
        logger.info('Seed: %d' % self.seed)

        fitting = ModelFitting(processor, self.parameters)

        settings = {
            'calibration_mode': self.calibration_mode,
            'generations': self.algorithm.generations,
            'population_size': self.algorithm.population_size,
            'simulation_output': self.output_type,
            'fitness_func': self.fitness_function,
            'target_output': self.target_data_path,
            'target_fit_range': self.target_fit_range,
            'out_fit_range': self.output_fit_range,
            'weighting': self.weighting_path,
            'champions_file': output_files[0],
            'population_file': output_files[1]
        }
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
