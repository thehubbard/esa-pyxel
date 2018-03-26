"""TBW."""
import logging
import importlib
from ast import literal_eval


def evaluate_reference(reference_str):
    """Evaluate a module's class, function, or constant.

    :param str reference_str: the python expression to
        evaluate or retrieve the module attribute reference to.
        This is usually a reference to a class or a function.
    :return: the module attribute or object.
    :rtype: object
    :raises ImportError: if reference_str cannot be evaulated to a callable.
    """
    if not reference_str:
        raise ImportError('Empty string cannot be evaluated')

    if '.' not in reference_str:
        raise ImportError('Missing module path')

    # reference to a module class, function, or constant
    module_str, function_str = reference_str.rsplit('.', 1)
    try:
        module = importlib.import_module(module_str)
    except ImportError as exc:
        raise ImportError('Cannot import module: %r. exc: %s' % (module_str, str(exc)))

    try:
        reference = getattr(module, function_str)
        # if isinstance(reference, type):
        #     # this is a class type, instantiate it using default arguments.
        #     reference = reference()
    except AttributeError:
        raise ImportError('Module: %s, does not contain %s' % (module_str, function_str))

    return reference


def eval_range(values):
    """Evaluate a string representation of a list or numpy array.

    :param values:
    :return: list
    """
    if values:
        # TODO: consider using this instead:
        # result = eval(value, {}, np.__dict__)
        if isinstance(values, str):
            if 'numpy' in values:
                locals_dict = {'numpy': importlib.import_module('numpy')}
                globals_dict = None
                values = eval(values, globals_dict, locals_dict)
                # NOTE: the following casting is to ensure JSON serialization works
                # JSON does not accept numpy.int* or numpy.float* types.
                if values.dtype == float:
                    values = [float(value) for value in values]
                elif values.dtype == int:
                    values = [int(value) for value in values]
                else:
                    logging.warning('numpy data type is not a float or int: %r', values)
            else:
                values = eval(values)

        if not isinstance(values, (list, tuple)):
            values = list(values)
    else:
        values = []
    return values


def eval_entry(value):
    """TBW.

    :param value:
    :return:
    """
    if isinstance(value, str):
        try:
            literal_eval(value)
        except (SyntaxError, ValueError, NameError):
            # ensure quotes in case of string literal value
            if value[0] == "'" and value[-1] == "'":
                pass
            elif value[0] == '"' and value[-1] == '"':
                pass
            else:
                value = '"' + value + '"'

        value = literal_eval(value)
    return value