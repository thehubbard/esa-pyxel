"""TBW."""
from pyxel.pipelines.validator import AttrClass
from pyxel.pipelines.validator import attr_class
from pyxel.pipelines.validator import attr_def
from pyxel.pipelines.validator import check_range


@attr_class
class Environment(AttrClass):
    """TBW."""

    temperature = attr_def(type=float, default=None, units='K', check=check_range(5.00, 400.0, 0.01))

    total_ionising_dose = attr_def(type=float, default=None, units='MeV/g')

    total_non_ionising_dose = attr_def(type=float, default=None, units='MeV/g')

    def copy(self):
        """TBW."""
        return Environment(**self.__getstate__())

    def __getstate__(self):
        """TBW.

        :return:
        """
        return {'temperature': self.temperature,
                'total_ionising_dose': self.total_ionising_dose,
                'total_non_ionising_dose': self.total_non_ionising_dose}

    # TODO: create unittests for this method
    def __eq__(self, obj):
        """TBW.

        :param obj:
        :return:
        """
        assert isinstance(obj, Environment)
        return self.__getstate__() == obj.__getstate__()


# class Environment:
#     """TBW."""
#
#     def __init__(self,
#                  temperature: float = None,
#                  total_ionising_dose: float = None,
#                  total_non_ionising_dose: float = None) -> None:
#         """TBW.
#
#         :param temperature:
#         :param total_ionising_dose:
#         :param total_non_ionising_dose:
#         """
#         self.temperature = temperature                              # unit: K
#         self.total_ionising_dose = total_ionising_dose              # unit: ?  (Gray or MeV/g) todo
#         self.total_non_ionising_dose = total_non_ionising_dose      # unit: ?  (Gray or MeV/g) todo
#
#     def copy(self):
#         """TBW."""
#         return Environment(**self.__getstate__())
#
#     def __getstate__(self):
#         """TBW.
#
#         :return:
#         """
#         return {'temperature': self.temperature,
#                 'total_ionising_dose': self.total_ionising_dose,
#                 'total_non_ionising_dose': self.total_non_ionising_dose}
#
#     # TODO: create unittests for this method
#     def __eq__(self, obj):
#         """TBW.
#
#         :param obj:
#         :return:
#         """
#         assert isinstance(obj, Environment)
#         return self.__getstate__() == obj.__getstate__()
