"""TBW."""
import logging
import numpy as np
try:
    import pygmo as pg
except ImportError:
    import warnings
    warnings.warn("Cannot import 'pygmo", RuntimeWarning, stacklevel=2)
from ..util.outputs import Outputs
from ..util import validators, config
from pyxel.calibration.fitting import ModelFitting
from pyxel.pipelines.model_function import ModelFunction
from pyxel.pipelines.processor import Processor


@config.detector_class
class Algorithm:
    """TBW.

    :return:
    """

    type = config.attribute(
        type=str,
        validator=[validators.validate_type(str),
                   validators.validate_choices(['sade', 'sga', 'nlopt'])],
        default='sade',
        doc=''
    )
    generations = config.attribute(
        type=int,
        validator=[validators.validate_type(int),
                   validators.validate_range(1, 100000)],
        default=1,
        doc=''
    )
    population_size = config.attribute(
        type=int,
        validator=[validators.validate_type(int),
                   validators.validate_range(1, 100000)],
        default=1,
        doc=''
    )

    # SADE #####
    variant = config.attribute(type=int, validator=[validators.validate_type(int),
                                                    validators.validate_range(1, 18)], default=2, doc='')
    variant_adptv = config.attribute(type=int, validator=[validators.validate_type(int),
                                                          validators.validate_range(1, 2)], default=1, doc='')
    ftol = config.attribute(type=float, default=1e-06, doc='')  # validator=validators.validate_range(),
    xtol = config.attribute(type=float, default=1e-06, doc='')  # validator=validators.validate_range(),
    memory = config.attribute(type=bool, default=False, doc='')
    # SADE #####

    # SGA #####
    cr = config.attribute(type=float, converter=float, validator=[validators.validate_type(float),
                                                                  validators.validate_range(0, 1)], default=0.9, doc='')
    eta_c = config.attribute(type=float, converter=float,
                             validator=[validators.validate_type(float)], default=1.0, doc='')
    m = config.attribute(type=float, converter=float, validator=[validators.validate_type(float),
                                                                 validators.validate_range(0, 1)], default=0.02, doc='')
    param_m = config.attribute(type=float, default=1.0, doc='')            # validator=validators.validate_range(1, 2),
    param_s = config.attribute(type=int, default=2, doc='')                # validator=validators.validate_range(1, 2),
    crossover = config.attribute(type=str, default='exponential', doc='')  # validator=validators.validate_choices(),
    mutation = config.attribute(type=str, default='polynomial', doc='')    # validator=validators.validate_choices(),
    selection = config.attribute(type=str, default='tournament', doc='')   # validator=validators.validate_choices(),
    # SGA #####

    # NLOPT #####
    nlopt_solver = config.attribute(type=str, default='neldermead',
                                    doc='')    # validator=validators.validate_choices(),  todo
    maxtime = config.attribute(type=int, default=0,
                               doc='')                     # validator=validators.validate_range(),  todo
    maxeval = config.attribute(type=int, default=0, doc='')
    xtol_rel = config.attribute(type=float, default=1.e-8, doc='')
    xtol_abs = config.attribute(type=float, default=0., doc='')
    ftol_rel = config.attribute(type=float, default=0., doc='')
    ftol_abs = config.attribute(type=float, default=0., doc='')
    stopval = config.attribute(type=float, default=float('-inf'), doc='')
    local_optimizer = config.attribute(type=None, default=None,
                                       doc='')          # validator=validators.validate_choices(),  todo
    replacement = config.attribute(type=str, default='best', doc='')
    nlopt_selection = config.attribute(type=str, default='best',
                                       doc='')         # todo: "selection" - same name as in SGA
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


@config.detector_class
class Calibration:
    """TBW.

    :return:
    """

    output_dir = None
    fitting = None          # type: ModelFitting
    calibration_mode = config.attribute(
        type=str,
        validator=[validators.validate_type(str),
                   validators.validate_choices(['pipeline', 'single_model'])],
        default='pipeline',
        doc=''
    )
    result_type = config.attribute(
        type=str,
        validator=[validators.validate_type(str),
                   validators.validate_choices(['image', 'signal', 'pixel'])],
        default='image',
        doc=''
    )
    result_fit_range = config.attribute(
        type=list,
        validator=[validators.validate_type(list)],
        default=None,
        doc=''
    )
    result_input_arguments = config.attribute(
        type=list,
        # validator=[validators.validate_type(list)],
        default=None,
        doc=''
    )
    target_data_path = config.attribute(
        type=list,
        validator=[validators.validate_type(list)],
        default=None,
        doc=''
    )
    target_fit_range = config.attribute(
        type=list,
        validator=[validators.validate_type(list)],
        default=None,
        doc=''
    )
    fitness_function = config.attribute(
        type=ModelFunction,
        validator=[validators.validate_type(ModelFunction)],
        default='',
        doc=''
    )
    algorithm = config.attribute(
        type=Algorithm,
        validator=[validators.validate_type(Algorithm)],
        default='',
        doc=''
    )
    parameters = config.attribute(
        type=list,
        validator=[validators.validate_type(list)],
        default='',
        doc=''
    )
    seed = config.attribute(
        type=int,
        validator=[validators.validate_type(int),
                   validators.validate_range(0, 100000)],
        default=np.random.randint(0, 100000),
        doc=''
    )
    islands = config.attribute(
        type=int,
        validator=[validators.validate_type(int),
                   validators.validate_range(0, 100)],
        default=0,
        doc=''
    )
    weighting_path = config.attribute(
        type=list,
        # validator=[validators.validate_type(list)],  # todo
        default=None,
        doc=''
    )

    def run_calibration(self, processor: Processor, output_dir: str = None):
        """TBW.

        :param processor: Processor object
        :param output_dir: Output object
        :return:
        """
        pg.set_global_rng_seed(seed=self.seed)
        logger = logging.getLogger('pyxel')
        logger.info('Seed: %d' % self.seed)

        self.output_dir = output_dir
        self.fitting = ModelFitting(processor, self.parameters)

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
            'file_path': output_dir
        }
        self.fitting.configure(settings)

        prob = pg.problem(self.fitting)
        opt_algorithm = self.algorithm.get_algorithm()
        algo = pg.algorithm(opt_algorithm)

        # try:
        #     bfe = pg.bfe(udbfe=pg.member_bfe())
        # except AttributeError:
        #     bfe = None
        #
        # try:
        #     archi = pg.archipelago(n=self.islands, algo=algo, prob=prob, b=bfe,
        #                            pop_size=self.algorithm.population_size, udi=pg.mp_island())
        # except KeyError:
        #     archi = pg.archipelago(n=self.islands, algo=algo, prob=prob,
        #                            pop_size=self.algorithm.population_size, udi=pg.mp_island())
        #
        # try:
        #     pop = pg.population(prob=prob, size=self.algorithm.population_size, b=bfe)
        # except TypeError:
        #     pop = pg.population(prob=prob, size=self.algorithm.population_size)

        if self.islands > 1:  # default
            archi = pg.archipelago(n=self.islands, algo=algo, prob=prob,
                                   pop_size=self.algorithm.population_size, udi=pg.mp_island())
            archi.evolve()
            logger.info(archi)
            archi.wait_check()
            champion_f = archi.get_champions_f()
            champion_x = archi.get_champions_x()
        else:
            pop = pg.population(prob=prob, size=self.algorithm.population_size)
            pop = algo.evolve(pop)
            champion_f = [pop.champion_f]
            champion_x = [pop.champion_x]

        self.fitting.file_matching_renaming()
        res = []  # type: list
        for f, x in zip(champion_f, champion_x):
            res += [self.fitting.get_results(overall_fitness=f, parameter=x)]

        logger.info('Calibration ended.')
        return res

    def post_processing(self, calib_results: list, output: Outputs):
        """TBW."""
        for item in calib_results:
            proc_list = item[0]
            result_dict = item[1]

            output.calibration_outputs(processor_list=proc_list)

            ii = 0
            for processor, target_data in zip(proc_list, self.fitting.all_target_data):
                simulated_data = self.fitting.get_simulated_data(processor)
                output.fitting_plot(target_data=target_data, simulated_data=simulated_data, data_i=ii)
                ii += 1
            output.fitting_plot_close(result_type=self.result_type, island=result_dict['island'])

        output.calibration_plots(calib_results[0][1])
