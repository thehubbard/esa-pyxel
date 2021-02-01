#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
#
#
from jinja2 import Template
import typing as t
if t.TYPE_CHECKING:
    from pyxel.pipelines import ModelGroup, ModelFunction


def html_display(obj: t.Callable) -> str:
    d = {key: str(obj.__dict__[key]).replace(">", "&gt").replace("<", "&lt") for key in obj.__dict__.keys()}

    template_str = """
    <h4>{{name}}</h4>
    <table>
    {% for key, value in data.items() %}
        <tr>
            <th> {{ key }} </th>
            <td> {{ value }} </td>
        </tr>
    {% endfor %}
    </table>"""

    template = Template(template_str)

    return template.render(data=d, name=obj.__class__.__name__)


def html_display_model(mf: "ModelFunction") -> str:
    d = {key: str(mf.__dict__[key]).replace(">", "&gt").replace("<", "&lt") for key in mf.__dict__.keys()}
    a = {key: str(mf._arguments[key]).replace(">", "&gt").replace("<", "&lt") for key in mf._arguments.keys()}

    template_str = """
    <h4>ModelFunction: {{model_name}}</h4>
    <table>
    {% for key, value in data.items() %}
        <tr>
            <th> {{ key }} </th>
            <td> {{ value }} </td>
        </tr>
    {% endfor %}
    </table>
    <b>Arguments</b>
    <table>
    {% for key, value in arguments.items() %}
        <tr>
            <th> {{ key }} </th>
            <td> {{ value }} </td>
        </tr>
    {% endfor %}
    </table>
    """

    template = Template(template_str)

    return template.render(data=d, arguments=a, model_name=mf._name)


def html_display_model_group(mg: "ModelGroup") -> str:

    d = {key: str(mg.__dict__[key]).replace(">", "&gt").replace("<", "&lt") for key in mg.__dict__.keys()}
    m = {model._name: f"{model!r}".replace(">", "&gt").replace("<", "&lt") for model in mg.models}

    template_str = """
    <h4>ModelGroup: {{model_group_name}}</h4>
    <table>
    {% for key, value in data.items() %}
        <tr>
            <th> {{ key }} </th>
            <td> {{ value }} </td>
        </tr>
    {% endfor %}
    </table>
    <b>Models</b>
    <table>
    {% for key, value in models.items() %}
        <tr>
            <th> {{ key }} </th>
            <td> {{ value }} </td>
        </tr>
    {% endfor %}
    </table>
    """

    template = Template(template_str)

    return template.render(data=d, models=m, model_group_name=mg._name)