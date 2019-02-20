"""Utility functions for creating outputs."""
import os
import glob
from copy import copy
from shutil import copy2
import typing as t          # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image
import astropy.io.fits as fits
try:
    import matplotlib.pyplot as plt
except ImportError:
    # raise Warning('Matplotlib cannot be imported')        # todo
    pass


class Outputs:
    """TBW."""

    def __init__(self,
                 output_folder: str = 'outputs',
                 parametric_plot: dict = None,
                 calibration_plot: dict = None,
                 single_plot: dict = None):
        """TBW."""
        self.input_file = None                      # type: t.Optional[str]
        self.champions_file = None                  # type: t.Optional[str]
        self.population_file = None                 # type: t.Optional[str]

        self.parametric_plot = parametric_plot      # type: t.Optional[dict]
        self.calibration_plot = calibration_plot    # type: t.Optional[dict]
        self.single_plot = single_plot              # type: t.Optional[dict]
        self.user_plt_args = None                   # type: t.Optional[dict]

        self.output_dir = apply_run_number(output_folder + '/run_??')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            raise IsADirectoryError('Directory exists.')

        self.default_ax_args = {
            'xlabel': None, 'ylabel': None, 'title': None, 'axis': None, 'grid': False,
            'xscale': 'linear', 'yscale': 'linear', 'xticks': None, 'yticks': None,
            'xlim': [None, None], 'ylim': [None, None],
        }   # type: dict
        self.default_plot_args = {
            'color': None, 'marker': '.', 'linestyle': ''
        }   # type: dict
        self.default_hist_args = {
            'bins': None, 'range': None, 'density': None, 'log': False, 'cumulative': False,
            'histtype': 'step', 'color': None, 'facecolor': None,
        }   # type: dict
        self.default_scatter_args = {
            'size': None, 'cbar_label': None
        }   # type: dict

        self.plt_args = None

        self.parameter_values = np.array([])
        self.parameter_keys = []                # type: list
        self.additional_keys = []               # type: list
        plt.figure()

    def set_input_file(self, filename: str):
        """TBW."""
        self.input_file = filename              # type: str
        copy2(self.input_file, self.output_dir)

    def create_files(self):
        """TBW."""
        self.champions_file = self.new_file('champions.out')        # type: str
        self.population_file = self.new_file('population.out')      # type: str
        return self.champions_file, self.population_file

    def new_file(self, filename: str):
        """TBW."""
        filename = self.output_dir + '/' + filename
        file = open(filename, 'wb')  # truncate output file
        file.close()
        return filename

    def save_to_bitmap(self, array, filename='image_??'):       # todo: finish, PIL does not work with JPEG !
        """Write array to bitmap PNG image file."""
        filename = self.output_dir + '/' + filename + '.PNG'
        filename = apply_run_number(filename)
        im = Image.fromarray(array)
        try:
            im.save(filename, "PNG")                    # todo: sometimes saving in PNG does not work too
        except OSError:
            pass
        # with this, it works: simple_digitization / numpy.uint32
        # with this, it does not work: simple_digitization / numpy.uint16

    def save_to_fits(self, array, filename='image_??'):
        """Write array to FITS file."""
        filename = self.output_dir + '/' + filename + '.fits'
        filename = apply_run_number(filename)
        hdu = fits.PrimaryHDU(array)
        hdu.writeto(filename, overwrite=False, output_verify='exception')

    def save_to_hdf(self, data, key='data', filename='store_??'):
        """Write object to HDF5 file."""
        filename = self.output_dir + '/' + filename + '.h5'
        filename = apply_run_number(filename)
        with pd.HDFStore(filename) as store:
            store[key] = data
        # todo: append more objects if needed to the same HDF5 file
        # todo: save whole detector object to a HDF5 file?

    def save_to_csv(self, dataframe: pd.DataFrame, filename='dataframe_??'):
        """Write pandas Dataframe to CSV file."""
        filename = self.output_dir + '/' + filename + '.csv'
        filename = apply_run_number(filename)
        dataframe.to_csv(filename, float_format='%g')

    def save_to_npy(self, array, filename='array_??'):
        """Write array to Numpy binary npy file."""
        filename = self.output_dir + '/' + filename + '.npy'
        filename = apply_run_number(filename)
        np.save(file=filename, arr=array)

    def save_plot(self, filename='figure_??'):
        """Save plot figure in PNG format, close figure and create new canvas for next plot."""
        filename = self.output_dir + '/' + filename + '.png'
        filename = apply_run_number(filename)
        plt.savefig(filename)
        plt.close('all')
        plt.figure()

    def plot_graph(self, x, y, args: dict = None):
        """TBW."""
        arg_tpl = self.update_args(plot_type='graph', new_args=args)
        ax_args, plt_args = self.update_args(plot_type='graph', new_args=self.user_plt_args, def_args=arg_tpl)
        plt.plot(x, y, color=plt_args['color'], marker=plt_args['marker'], linestyle=plt_args['linestyle'])
        update_plot(ax_args)
        plt.draw()

    def plot_histogram(self, data, args: dict = None):
        """TBW."""
        arg_tpl = self.update_args(plot_type='hist', new_args=args)
        ax_args, plt_args = self.update_args(plot_type='hist', new_args=self.user_plt_args, def_args=arg_tpl)
        if isinstance(data, np.ndarray):
            data = data.flatten()
        plt.hist(x=data,
                 bins=plt_args['bins'], range=plt_args['range'],
                 density=plt_args['density'], log=plt_args['log'], cumulative=plt_args['cumulative'],
                 histtype=plt_args['histtype'], color=plt_args['color'], facecolor=plt_args['facecolor'])
        update_plot(ax_args)
        plt.draw()

    def plot_scatter(self, x, y, color=None, args: dict = None):
        """TBW."""
        arg_tpl = self.update_args(plot_type='scatter', new_args=args)
        ax_args, plt_args = self.update_args(plot_type='scatter', new_args=self.user_plt_args, def_args=arg_tpl)
        fig = plt.gcf()
        if color is not None:
            sp = plt.scatter(x, y, c=color, s=plt_args['size'])
            cbar = fig.colorbar(sp)
            cbar.set_label(plt_args['cbar_label'])
        else:
            plt.scatter(x, y, s=plt_args['size'])
        update_plot(ax_args)
        plt.draw()

    def single_output(self, processor):
        """TBW."""
        self.save_to_fits(array=processor.detector.image.array)
        # self.save_to_bitmap(array=processor.detector.image.array)
        # self.save_to_npy(array=processor.detector.image.array)                          # todo
        # self.save_to_hdf(data=processor.detector.charges.frame, key='charge')
        # self.save_to_csv(dataframe=processor.detector.charges.frame)

        self.user_plt_args = None
        x = processor.detector.photons.array                    # todo: default plots with plot_args?
        y = processor.detector.image.array
        color = None
        if self.single_plot:
            if 'plot_args' in self.single_plot:
                self.user_plt_args = self.single_plot['plot_args']
            if 'x' in self.single_plot:
                x = processor.get(self.single_plot['x'])
            if 'y' in self.single_plot:
                y = processor.get(self.single_plot['y'])
            if 'plot_type' in self.single_plot:
                if isinstance(x, np.ndarray):
                    x = x.flatten()
                if isinstance(y, np.ndarray):
                    y = y.flatten()

                if self.single_plot['plot_type'] == 'graph':
                    self.plot_graph(x, y)
                    fname = 'graph_??'
                elif self.single_plot['plot_type'] == 'histogram':
                    self.plot_histogram(y)
                    fname = 'histogram_??'
                elif self.single_plot['plot_type'] == 'scatter':
                    self.plot_scatter(x, y, color)
                    fname = 'scatter_??'
                else:
                    raise KeyError()
                self.save_to_npy(x, 'x_'+fname)
                self.save_to_npy(y, 'y_'+fname)
                self.save_plot(fname)

    def champions_plot(self, results):
        """TBW."""
        data = np.loadtxt(self.champions_file)
        generations = data[:, 0].astype(int)
        title = 'Calibrated parameter: '
        items = list(results.items())
        a = 1
        for item in items:
            plt_args = {'xlabel': 'generation', 'linestyle': '-'}
            key = item[0]
            param_value = item[1]
            param_name = key[key.rfind('.') + 1:]
            plt_args['ylabel'] = param_name
            if param_name == 'fitness':
                plt_args['title'] = 'Champion fitness'
                plt_args['color'] = 'red'
            else:
                if key.rfind('.arguments') == -1:
                    mdn = key[:key.rfind('.' + param_name)]
                else:
                    mdn = key[:key.rfind('.arguments')]
                model_name = mdn[mdn.rfind('.') + 1:]
                plt_args['title'] = title + model_name + ' / ' + param_name

            b = 1
            if isinstance(param_value, float) or isinstance(param_value, int):
                column = data[:, a]
                self.plot_graph(generations, column, args=plt_args)
            elif isinstance(param_value, np.ndarray):
                b = len(param_value)
                column = data[:, a:a + b]
                self.plot_graph(generations, column, args=plt_args)
                plt.legend(range(b))

            self.save_plot('calibrated_parameter_??')
            a += b

    def population_plot(self):
        """TBW."""
        data = np.loadtxt(self.population_file)
        fitnesses = np.log10(data[:, 1])
        a, b = 2, 1                 # 1st parameter and fitness
        if self.calibration_plot['population_plot']:
            if 'columns' in self.calibration_plot['population_plot']:
                col = self.calibration_plot['population_plot']['columns']
                a, b = col[0], col[1]
        x = data[:, a]
        y = data[:, b]
        plt_args = {'xlabel': 'calibrated parameter', 'ylabel': 'fitness',
                    'title': 'Population of the last generation',
                    'size': 8, 'cbar_label': 'log(fitness)'}
        if a == 1 or b == 1:
            self.plot_scatter(x, y, args=plt_args)
        else:
            self.plot_scatter(x, y, color=fitnesses, args=plt_args)
        self.save_plot('population_??')

    def calibration_output(self, processor, results: dict):
        """TBW."""
        self.single_output(processor)

        if self.calibration_plot:
            if 'champions_plot' in self.calibration_plot:
                self.user_plt_args = None
                if self.calibration_plot['champions_plot']:
                    if 'plot_args' in self.calibration_plot['champions_plot']:
                        self.user_plt_args = self.calibration_plot['champions_plot']['plot_args']
                self.champions_plot(results)
            if 'population_plot' in self.calibration_plot:
                self.user_plt_args = None
                if self.calibration_plot['population_plot']:
                    if 'plot_args' in self.calibration_plot['population_plot']:
                        self.user_plt_args = self.calibration_plot['population_plot']['plot_args']
                self.population_plot()

    def add_parametric_step(self, parametric, processor):
        """TBW."""
        # self.single_output(processor.detector)

        row = np.array([])
        for var in parametric.enabled_steps:
            row = np.append(row, processor.get(var.key))
            if var.key not in self.parameter_keys:
                self.parameter_keys += [var.key]

        additional_params = [self.parametric_plot['x'], self.parametric_plot['y']]
        for key in additional_params:
            if key is not None and key not in self.parameter_keys:
                row = np.append(row, processor.get(key))
                if key not in self.additional_keys:
                    self.additional_keys += [key]

        if self.parameter_values.size == 0:
            self.parameter_values = row
        else:
            self.parameter_values = np.vstack((self.parameter_values, row))

    def update_args(self, plot_type: str, new_args: dict = None, def_args: tuple = (None, None)):
        """TBW."""
        ax_args, plt_args = def_args[0], def_args[1]
        if ax_args is None:
            ax_args = copy(self.default_ax_args)
        if plt_args is None:
            if plot_type == 'hist':
                plt_args = copy(self.default_hist_args)
            elif plot_type == 'graph':
                plt_args = copy(self.default_plot_args)
            elif plot_type == 'scatter':
                plt_args = copy(self.default_scatter_args)
            else:
                raise ValueError

        if new_args:
            for key in new_args:
                if key in plt_args.keys():
                    plt_args[key] = new_args[key]
                elif key in ax_args.keys():
                    ax_args[key] = new_args[key]
                else:
                    raise KeyError('Not valid plotting key in "plot_args": "%s"' % key)

        return ax_args, plt_args

    def parametric_output(self):
        """TBW."""
        self.parameter_keys += self.additional_keys
        self.user_plt_args = None

        if self.parametric_plot:
            if 'x' in self.parametric_plot:
                x_key = self.parametric_plot['x']
            else:
                raise KeyError()    # x_key = self.parameter_keys[0]
            if 'y' in self.parametric_plot:
                y_key = self.parametric_plot['y']
            else:
                raise KeyError()
            if 'plot_args' in self.parametric_plot:
                self.user_plt_args = self.parametric_plot['plot_args']
        else:
            raise KeyError()

        x = self.parameter_values[:, self.parameter_keys.index(x_key)]
        y = self.parameter_values[:, self.parameter_keys.index(y_key)]

        par_name = x_key[x_key.rfind('.') + 1:]
        res_name = y_key[y_key.rfind('.') + 1:]
        args = {'xlabel': par_name, 'ylabel': res_name}

        if isinstance(x, np.ndarray):
            x = x.flatten()
        if isinstance(y, np.ndarray):
            y = y.flatten()

        self.plot_graph(x, y, args=args)
        self.save_to_npy(x, 'x_parametric_??')
        self.save_to_npy(y, 'y_parametric_??')
        self.save_plot('parametric_??')


def show_plots():
    """Close last empty canvas and show all the previously created figures."""
    plt.close()
    plt.show()


def update_plot(ax_args):
    """TBW."""
    plt.xlabel(ax_args['xlabel'])
    plt.ylabel(ax_args['ylabel'])
    plt.xscale(ax_args['xscale'])
    plt.yscale(ax_args['yscale'])
    plt.xlim(ax_args['xlim'][0], ax_args['xlim'][1])
    plt.ylim(ax_args['ylim'][0], ax_args['ylim'][1])
    plt.title(ax_args['title'])
    if ax_args['axis']:
        plt.axis(ax_args['axis'])
    if ax_args['xticks']:
        plt.xticks(ax_args['xticks'])
    if ax_args['yticks']:
        plt.yticks(ax_args['yticks'])
    plt.grid(ax_args['grid'])


def update_fits_header(header, key, value):
    """TBW.

    :param header:
    :param key:
    :param value:
    :return:
    """
    if not isinstance(value, (str, int, float)):
        value = '%r' % value
    if isinstance(value, str):
        value = value[0:24]
    if isinstance(key, (list, tuple)):
        key = '/'.join(key)
    key = key.replace('.', '/')[0:36]
    header[key] = value


def apply_run_number(path):
    """Convert the file name numeric placeholder to a unique number.

    :param path:
    :return:
    """
    path_str = str(path)
    if '?' in path_str:
        dir_list = sorted(glob.glob(path_str))
        p_0 = path_str.find('?')
        p_1 = path_str.rfind('?')
        template = path_str[p_0: p_1 + 1]
        path_str = path_str.replace(template, '{:0%dd}' % len(template))
        last_num = 0
        if len(dir_list):
            path_last = dir_list[-1]
            last_num = int(path_last[p_0: p_1 + 1])
        last_num += 1
        path_str = path_str.format(last_num)
    return type(path)(path_str)