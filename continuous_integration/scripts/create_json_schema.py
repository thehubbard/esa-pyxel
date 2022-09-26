"""Create JSON Schema."""

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import functools
import importlib
import inspect
import textwrap
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

import click
from boltons.strutils import under2camel
from numpydoc.docscrape import NumpyDocString
from toolz import dicttoolz
from tqdm.auto import tqdm

from pyxel import __version__


@dataclass
class Param:
    description: str
    annotation: t.Any


@dataclass
class ParamDefault:
    description: str
    annotation: t.Any
    default: t.Any


@dataclass
class FuncDocumentation:
    description: str
    parameters: t.Mapping[str, t.Union[Param, ParamDefault]]


@dataclass
class ModelInfo:
    model_name: str
    model_fullname: str
    model_class_name: str
    func: t.Callable


@dataclass
class ModelGroupInfo:
    name: str
    class_name: str


@dataclass(frozen=True)
class Klass:
    cls: t.Type
    base_cls: t.Optional[t.Type] = None

    @property
    def name(self) -> str:
        return self.cls.__name__


def get_documentation(func: t.Callable) -> FuncDocumentation:
    assert func.__doc__
    doc = NumpyDocString(inspect.cleandoc(func.__doc__))

    signature = inspect.signature(func)  # type: inspect.Signature

    parameters = {}

    if doc["Parameters"]:
        for params in doc["Parameters"]:
            name, *_ = params.name.split(":")
            description = "\n".join(params.desc)

            parameter = signature.parameters[name]  # type: inspect.Parameter

            if t.get_origin(parameter.annotation):
                annotation = str(parameter.annotation)  # type: str
            elif hasattr(parameter.annotation, "__name__"):
                annotation = parameter.annotation.__name__
            else:
                annotation = str(parameter.annotation)

            if parameter.default != inspect.Parameter.empty:
                param = ParamDefault(
                    description=description,
                    annotation=annotation.replace("NoneType", "None"),
                    default=parameter.default,
                )  # type: t.Union[ParamDefault, Param]
            else:
                param = Param(description=description, annotation=annotation)

            parameters[name.strip()] = param
    else:
        for name, parameter in signature.parameters.items():
            if hasattr(parameter.annotation, "__name__"):
                annotation = parameter.annotation.__name__  # type: str
            else:
                annotation = str(parameter.annotation)

            if parameter.default != inspect.Parameter.empty:
                param = ParamDefault(
                    description="",
                    annotation=annotation.replace("NoneType", "None"),
                    default=parameter.default,
                )  # type: t.Union[ParamDefault, Param]
            else:
                param = Param(description="", annotation=annotation)

            parameters[name.strip()] = param

    return FuncDocumentation(
        description="\n".join(doc["Summary"]), parameters=parameters
    )


@functools.cache
def get_doc_from_klass(klass: Klass) -> FuncDocumentation:
    if klass.base_cls is None:
        doc = get_documentation(klass.cls)  # type: FuncDocumentation
    else:
        doc_base = get_documentation(klass.base_cls)  # type: FuncDocumentation
        doc_inherited = get_documentation(klass.cls)  # type: FuncDocumentation

        doc = FuncDocumentation(
            description=doc_inherited.description,
            parameters=dicttoolz.dissoc(doc_inherited.parameters, *doc_base.parameters),
        )

    return doc


def generate_class(klass: Klass) -> t.Iterator[str]:

    doc = get_doc_from_klass(klass)
    klass_description_lst = textwrap.wrap(doc.description)  # type: t.Sequence[str]

    yield "@schema("
    if len(klass_description_lst) == 1:
        yield f"    title={klass.name!r},"
        yield f"    description={klass_description_lst[0]!r}"
    elif len(klass_description_lst) > 1:
        yield f"    title={klass.name!r},"
        yield "    description=("
        for line in klass_description_lst:
            yield f"        {line!r}"
        yield "        )"
    else:
        yield f"    title={klass.name!r}"

    yield ")"
    yield "@dataclass"

    if klass.base_cls is None:
        yield f"class {klass.cls.__name__}:"
    else:
        yield f"class {klass.cls.__name__}({klass.base_cls.__name__}):"

    if doc.parameters:
        for name, param in doc.parameters.items():
            title = name

            yield f"    {name}: {param.annotation} = field("
            if isinstance(param, ParamDefault):
                yield f"        default={param.default!r},"
            yield "        metadata=schema("

            description_lst = textwrap.wrap(param.description)  # type: t.Sequence[str]
            if len(description_lst) == 1:
                yield f"            title={title!r},"
                yield f"            description={description_lst[0]!r}"
            elif len(description_lst) > 1:
                yield f"            title={title!r},"
                yield "            description=("
                for line in description_lst:
                    yield f"                    {line!r}"
                yield "                )"
            else:
                yield f"            title={title!r}"

            yield "        )"
            yield "    )"

    else:
        yield "    pass"

    yield ""
    yield ""


def generate_model(
    func: t.Callable,
    func_name: str,
    func_fullname: str,
    model_name: str,
) -> t.Iterator[str]:

    doc = get_documentation(func)  # type: FuncDocumentation

    yield "@schema(title='Parameters')"
    yield "@dataclass"
    yield f"class {model_name}Arguments(typing.Mapping[str, typing.Any]):"

    dct = {key: value for key, value in doc.parameters.items() if key != "detector"}

    all_defaults = all(
        [isinstance(el, ParamDefault) for el in dct.values()]
    )  # type: bool

    if dct:
        for name, param in dct.items():
            title = name

            yield f"    {name}: {param.annotation} = field("
            if isinstance(param, ParamDefault):
                yield f"        default={param.default!r},"
            yield "        metadata=schema("
            yield f"            title={title!r}"

            description_lst = textwrap.wrap(param.description)  # type: t.Sequence[str]
            if len(description_lst) == 1:
                yield f"            ,description={description_lst[0]!r}"
            elif len(description_lst) > 1:
                yield "            ,description=("
                for line in description_lst:
                    yield f"                    {line!r}"
                yield "                )"

            yield "        )"
            yield "    )"
    # else:
    #     yield "    pass"

    yield "    def __iter__(self) -> typing.Iterator[str]:"
    yield f"        return iter({tuple(dct)!r})"

    yield "    def __getitem__(self, item: typing.Any) -> typing.Any:"
    yield "        if item in tuple(self):"
    yield "            return getattr(self, item)"
    yield "        else:"
    yield "            raise KeyError"

    yield "    def __len__(self) -> int:"
    yield f"        return {len(dct)}"

    yield ""
    yield ""
    yield "@schema("
    yield f"    title=\"Model '{func_name}'\""

    description_lst = textwrap.wrap(doc.description)
    if len(description_lst) == 1:
        yield f"    ,description={description_lst[0]!r}"
    elif len(description_lst) > 1:
        yield "    ,description=("
        for line in description_lst:
            yield f"        {line!r}"
        yield "    )"

    yield ")"
    yield "@dataclass"
    yield f"class {model_name}:"
    yield "    name: str"

    if all_defaults:
        yield f"    arguments: {model_name}Arguments = field(default_factory={model_name}Arguments)"
    else:
        yield f"    arguments: {model_name}Arguments"
    yield f"    func: typing.Literal[{func_fullname!r}] = {func_fullname!r}"
    yield "    enabled: bool = True"
    yield ""
    yield ""


def get_model_info(group_name: str) -> t.Sequence[ModelInfo]:
    group_module = importlib.import_module(f"pyxel.models.{group_name}")

    lst = []  # type: t.List[ModelInfo]
    for name in dir(group_module):
        if name.startswith("__"):
            continue

        func = getattr(group_module, name)  # type: t.Callable
        if not callable(func):
            continue

        sig = inspect.signature(func)  # type: inspect.Signature

        if "detector" not in sig.parameters:
            continue

        lst.append(
            ModelInfo(
                model_name=name,
                model_fullname=f"pyxel.models.{group_name}.{name}",
                model_class_name=under2camel(f"Model_{group_name}_{name}"),
                func=func,
            )
        )

    return lst


def capitalize_title(name: str) -> str:
    return " ".join([el.capitalize() for el in name.split("_")])


def generate_group(model_groups_info: t.Sequence[ModelGroupInfo]) -> t.Iterator[str]:

    for group_info in model_groups_info:
        group_name = group_info.name

        models_info = get_model_info(group_name)  # type: t.Sequence[ModelInfo]

        for info in models_info:  # type: ModelInfo

            group_name_title = capitalize_title(group_name)
            model_title = capitalize_title(info.model_name)

            yield "#"
            yield f"# Model: {group_name_title} / {model_title}"
            yield "#"

            yield from generate_model(
                func=info.func,
                func_name=info.model_name,
                func_fullname=info.model_fullname,
                model_name=info.model_class_name,
            )

    yield "#"
    yield "# Detection pipeline"
    yield "#"
    yield "@dataclass"
    yield "class DetailedDetectionPipeline(DetectionPipeline):"

    for group_info in model_groups_info:
        group_name = group_info.name
        group_name_title = capitalize_title(group_name)

        models_info = get_model_info(group_name)
        models_class_names = [info.model_class_name for info in models_info]

        all_model_class_names = ", ".join([*models_class_names, "ModelFunction"])

        yield f"    {group_name}: typing.Sequence["
        yield "        typing.Union["
        yield f"            {all_model_class_names}"
        yield "        ]"
        yield f"    ] = field(default_factory=list, metadata=schema(title={group_name_title!r}))"


def get_model_group_info() -> t.Sequence[ModelGroupInfo]:
    all_group_models = (
        "photon_generation",
        "optics",
        "phasing",
        "charge_generation",
        "charge_collection",
        "charge_measurement",
        "readout_electronics",
        "charge_transfer",
        "signal_transfer",
    )

    lst = []  # type: t.List[ModelGroupInfo]

    for group_name in all_group_models:
        lst.append(
            ModelGroupInfo(
                name=group_name, class_name=under2camel(f"model_group_{group_name}")
            )
        )

    return lst


@functools.cache
def create_klass(cls: t.Union[t.Type, str]) -> Klass:
    import pyxel.detectors

    if isinstance(cls, str):
        cls_type = getattr(pyxel.detectors, cls)  # type: t.Type
        return create_klass(cls_type)

    # Try to find a base class
    _, *base_classes, _ = inspect.getmro(cls)  # type: tuple

    if base_classes:
        return Klass(cls, base_cls=base_classes[0])

    return Klass(cls)


def create_graph(cls: t.Type, graph: t.Mapping[Klass, t.Set[Klass]]) -> None:
    klass = create_klass(cls)  # type: Klass

    doc = get_doc_from_klass(klass)  # type: FuncDocumentation

    if klass.base_cls is None:
        parameters = (
            doc.parameters
        )  # type: t.Mapping[str, t.Union[Param, ParamDefault]]
    else:
        create_graph(cls=klass.base_cls, graph=graph)
        klass_base = create_klass(klass.base_cls)  # type: Klass
        graph[klass].add(klass_base)

        klass_base_doc = get_doc_from_klass(klass_base)  # type:  FuncDocumentation

        parameters = {**doc.parameters, **klass_base_doc.parameters}

    for name, parameter in parameters.items():
        assert parameter.annotation

        klass_param = create_klass(parameter.annotation)  # type: Klass
        graph[klass].add(klass_param)

        if klass_param.base_cls is not None:
            klass_param_base = create_klass(klass_param.base_cls)  # type: Klass
            graph[klass_param].add(klass_param_base)


def generate_detectors() -> t.Iterator[str]:
    from pyxel.detectors import APD, CCD, CMOS, MKID

    registered_detectors = (CCD, CMOS, MKID, APD)  # type: t.Sequence[t.Type[Detector]]

    # Build a dependency graph
    graph = defaultdict(set)  # type: t.Mapping[Klass, t.Set[Klass]]
    for detector in registered_detectors:  # type: t.Type[Detector]
        create_graph(cls=detector, graph=graph)

    # Generate code based on the dependency graph
    ts = TopologicalSorter(graph)
    for klass in ts.static_order():
        yield from generate_class(klass)

    yield ""
    yield "#"
    yield "# Outputs"
    yield "#"
    yield "ValidName = typing.Literal["
    yield "    'detector.image.array', 'detector.signal.array', 'detector.pixel.array'"
    yield "]"
    yield "ValidFormat = typing.Literal['fits', 'hdf', 'npy', 'txt', 'csv', 'png']"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class Outputs:"
    yield "    output_folder: pathlib.Path"
    yield "    save_data_to_file: typing.Optional["
    yield "        typing.Sequence[typing.Mapping[ValidName, typing.Sequence[ValidFormat]]]"
    yield "    ] = None"
    yield ""
    yield ""
    yield "#"
    yield "# Exposure"
    yield "#"
    yield "@dataclass"
    yield "class ExposureOutputs(Outputs):"
    yield "    save_exposure_data: typing.Optional["
    yield "        typing.Sequence[typing.Mapping[str, typing.Sequence[str]]]"
    yield "    ] = None"

    yield "@dataclass"
    yield "class Readout:"
    yield "    times: typing.Optional[typing.Union[typing.Sequence, str]] = None"
    yield "    times_from_file: typing.Optional[str] = None"
    yield "    start_time: float = 0.0"
    yield "    non_destructive: bool = False"
    yield ""
    yield ""
    yield "@schema(title='Exposure')"
    yield "@dataclass"
    yield "class Exposure:"
    yield "    outputs: ExposureOutputs"
    yield "    readout: Readout = field(default_factory=Readout)"
    yield "    result_type: typing.Literal['image', 'signal', 'pixel', 'all'] = 'all'"
    yield "    pipeline_seed: typing.Optional[int] = None"
    yield ""
    yield ""
    yield "#"
    yield "# Observation"
    yield "#"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class ObservationOutputs(Outputs):"
    yield "    save_observation_data: typing.Optional["
    yield "        typing.Sequence[typing.Mapping[str, typing.Sequence[str]]]"
    yield "    ] = None"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class ParameterValues:"
    yield "    key: str"
    yield "    values: typing.Union["
    yield "        typing.Literal['_'],"
    yield "        typing.Sequence[typing.Literal['_']],"
    yield "        typing.Sequence[typing.Union[int, float]],"
    yield "        typing.Sequence[str],"
    yield "    ]"
    yield "    boundaries: typing.Optional[typing.Tuple[float, float]] = None"
    yield "    enabled: bool = True"
    yield "    logarithmic: bool = False"
    yield ""
    yield ""
    yield "@schema(title='Observation')"
    yield "@dataclass"
    yield "class Observation:"
    yield "    outputs: ObservationOutputs"
    yield "    parameters: typing.Sequence[ParameterValues]"
    yield "    readout: Readout"
    yield "    mode: str = 'product'"
    yield "    from_file: typing.Optional[str] = None"
    yield "    column_range: typing.Optional[typing.Tuple[int, int]] = None"
    yield "    with_dask: bool = False"
    yield "    result_type: typing.Literal['image', 'signal', 'pixel', 'all'] = 'all'"
    yield "    pipeline_seed: typing.Optional[int] = None"
    yield ""
    yield ""
    yield "#"
    yield "# Calibration"
    yield "#"
    yield "@dataclass"
    yield "class CalibrationOutputs(Outputs):"
    yield "    save_calibration_data: typing.Optional["
    yield "        typing.Sequence[typing.Mapping[str, typing.Sequence[str]]]"
    yield "    ] = None"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class Algorithm:"
    yield "    type: typing.Literal['sade', 'sga', 'nlopt'] = 'sade'"
    yield "    generations: int = 1"
    yield "    population_size: int = 1"

    yield "    # SADE #####"
    yield "    variant: int = 2"
    yield "    variant_adptv: int = 1"
    yield "    ftol: float = 1e-6"
    yield "    xtol: float = 1e-6"
    yield "    memory: bool = False"

    yield "    # SGA #####"
    yield "    cr: float = 0.9"
    yield "    eta_c: float = 1.0"
    yield "    m: float = 0.02"
    yield "    param_m: float = 1.0"
    yield "    param_s: int = 2"
    yield "    crossover: typing.Literal['single', 'exponential', 'binominal', 'sbx'] = 'exponential'"
    yield "    mutation: typing.Literal['uniform', 'gaussian', 'polynomial'] = 'polynomial'"
    yield "    selection: typing.Literal['tournament', 'truncated'] = 'tournament'"

    yield "    # NLOPT #####"
    yield "    nlopt_solver: typing.Literal["
    yield "        'cobyla', 'bobyqa', 'newuoa', 'newuoa_bound', 'praxis', 'neldermead',"
    yield "        'sbplx', 'mma', 'ccsaq', 'slsqp', 'lbfgs', 'tnewton_precond_restart',"
    yield "        'tnewton_precond', 'tnewton_restart', 'tnewton', 'var2', 'var1', 'auglag',"
    yield "        'auglag_eq'"
    yield "    ] = 'neldermead'"
    yield "    maxtime: int = 0"
    yield "    maxeval: int = 0"
    yield "    xtol_rel: float = 1.0e-8"
    yield "    xtol_abs: float = 0.0"
    yield "    ftol_rel: float = 0.0"
    yield "    ftol_abs: float = 0.0"
    yield "    stopval: typing.Optional[float] = None"
    yield "    # local_optimizer: typing.Optional['pg.nlopt'] = None"
    yield "    replacement: typing.Literal['best', 'worst', 'random'] = 'best'"
    yield "    nlopt_selection: typing.Literal['best', 'worst', 'random'] = 'best'"
    yield ""
    yield ""
    yield "@schema(title='Fitness function')"
    yield "@dataclass"
    yield "class FitnessFunction:"
    yield "    func: str"
    yield ""
    yield ""
    yield "@schema(title='Calibration')"
    yield "@dataclass"
    yield "class Calibration:"
    yield "    outputs: CalibrationOutputs"
    yield "    target_data_path: typing.Sequence[pathlib.Path]"
    yield "    fitness_function: FitnessFunction"
    yield "    algorithm: Algorithm"
    yield "    parameters: typing.Sequence[ParameterValues]"
    yield "    readout: typing.Optional[Readout] = None"
    yield "    mode: typing.Literal['pipeline', 'single_model'] = 'pipeline'"
    yield "    result_type: typing.Literal['image', 'signal', 'pixel'] = 'image'"
    yield "    result_fit_range: typing.Optional[typing.Sequence[int]] = None"
    yield "    result_input_arguments: typing.Optional[typing.Sequence[ParameterValues]] = None"
    yield "    target_fit_range: typing.Optional[typing.Sequence[int]] = None"
    yield "    pygmo_seed: typing.Optional[int] = None"
    yield "    pipeline_seed: typing.Optional[int] = None"
    yield "    num_islands: int = 1"
    yield "    num_evolutions: int = 1"
    yield "    num_best_decisions: typing.Optional[int] = None"
    yield "    topology: typing.Literal['unconnected', 'ring', 'fully_connected'] = 'unconnected'"
    yield "    type_islands: typing.Literal["
    yield "        'multiprocessing', 'multithreading', 'ipyparallel'"
    yield "    ] = 'multiprocessing'"
    yield "    weights_from_file: typing.Optional[typing.Sequence[pathlib.Path]] = None"
    yield "    weights: typing.Optional[typing.Sequence[float]] = None"

    yield ""
    yield ""
    yield ""

    # Create wrappers for the detectors
    detector_classes = ["CCD", "CMOS", "MKID", "APD"]  # type: t.Sequence[str]
    # for klass_name in detector_classes:
    #     yield f"#@schema(title='{klass_name}')"
    #     yield "#@dataclass"
    #     yield f"#class Wrapper{klass_name}:"
    #     yield f"#    {klass_name.lower()}: {klass_name}"
    #     yield ""
    #     yield ""

    # Create wrappers for the modes
    mode_classes = ["Exposure", "Observation", "Calibration"]  # type: t.Sequence[str]
    # for klass_name in mode_classes:
    #     yield f"#@schema(title='{klass_name}')"
    #     yield "#@dataclass"
    #     yield f"#class Wrapper{klass_name}:"
    #     yield f"#    {klass_name.lower()}: {klass_name}"
    #     yield ""
    #     yield ""

    # wrapper_detector_classes = [f"Wrapper{el}" for el in detector_classes]
    # wrapper_mode_classes = [f"Wrapper{el}" for el in mode_classes]

    yield "@dataclass"
    yield "class Configuration:"
    yield "    pipeline: DetailedDetectionPipeline"
    # yield f"    # mode: typing.Union[{', '.join(wrapper_mode_classes)}]"
    # yield f"    # detector: typing.Union[{', '.join(wrapper_detector_classes)}]"

    yield ""
    yield "    # Running modes"
    for klass_name in mode_classes:
        yield f"    {klass_name.lower()}: typing.Optional[{klass_name}] = field(default=None, metadata=schema(title={klass_name!r}))"

    yield ""
    yield "    # Detectors"
    for klass_name in detector_classes:
        yield f"    {klass_name.lower()}_detector: typing.Optional[{klass_name}] = field(default=None, metadata=schema(title={klass_name!r}))"


def generate_all_models() -> t.Iterator[str]:
    lst = get_model_group_info()
    yield "#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021, 2022."
    yield "#"
    yield "#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which"
    yield "#  is part of this Pyxel package. No part of the package, including"
    yield "#  this file, may be copied, modified, propagated, or distributed except according to"
    yield "#  the terms contained in the file ‘LICENCE.txt’."
    yield ""
    yield "######################################"
    yield "# Note: This code is auto-generated. #"
    yield "#       Don't modify it !            #"
    yield "######################################"
    yield ""

    yield "import pathlib"
    yield "import typing"
    yield "from dataclasses import dataclass, field"
    yield ""
    yield "import click"
    yield "from apischema import schema"
    yield ""
    yield ""

    yield "@dataclass"
    yield "class ModelFunction:"
    yield "    name: str"
    yield "    func: str = field(metadata=schema(pattern='^(?!pyxel\\.models\\.)'))"
    yield "    arguments: typing.Optional[typing.Mapping[str, typing.Any]] = None"
    yield "    enabled: bool = True"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class ModelGroup:"
    yield "    models: typing.Sequence[ModelFunction]"
    yield "    name: str"
    yield ""
    yield ""
    yield "@dataclass"
    yield "class DetectionPipeline:"
    yield "    photon_generation: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    optics: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    phasing: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    charge_generation: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    charge_collection: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    charge_measurement: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    readout_electronics: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    charge_transfer: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield "    signal_transfer: typing.Sequence[ModelFunction] = field(default_factory=list)"
    yield ""
    yield ""

    yield from generate_group(lst)
    yield from generate_detectors()

    yield ""
    yield ""
    yield "@click.command()"
    yield "@click.option("
    yield "    '-f',"
    yield "    '--filename',"
    yield "    default='../../static/pyxel_schema.json',"
    yield "    type=click.Path(),"
    yield "    help='JSON schema filename',"
    yield "    show_default=True,"
    yield ")"
    yield "def create_json_schema(filename: pathlib.Path):"
    yield "    import json"
    yield ""
    yield "    from apischema.json_schema import JsonSchemaVersion, deserialization_schema"
    yield "    dct_schema = deserialization_schema("
    yield "        Configuration, version=JsonSchemaVersion.DRAFT_7, all_refs=True,"
    yield "    )"
    yield ""
    yield "    print(json.dumps(dct_schema))"
    yield ""
    yield "    full_filename = pathlib.Path(filename).resolve()"
    yield ""
    yield "    with full_filename.open('w') as fh:"
    yield "        json.dump(obj=dct_schema, fp=fh, indent=2)"
    yield ""
    yield ""
    yield "if __name__ == '__main__':"
    yield "    create_json_schema()"


def create_auto_generated(filename: Path) -> None:
    """Create an auto-generated file."""
    with Path(filename).open("w") as fh:
        for line in tqdm(generate_all_models()):
            fh.write(f"{line}\n")


@click.command()
@click.option(
    "-f",
    "--filename",
    default="./auto_generated.py",
    type=click.Path(),
    help="Auto generated filename.",
    show_default=True,
)
@click.version_option(version=__version__)
def main(filename: Path):
    """Create an auto-generated file."""
    create_auto_generated(filename)


if __name__ == "__main__":
    main()
