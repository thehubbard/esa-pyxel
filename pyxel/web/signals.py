"""TBW."""

# import typing as t

from esapy_dispatcher import DispatcherBlocking
from esapy_dispatcher.dispatcher_meta import from_ref


HK_SIGNAL = 'HK_SIGNAL'
DATA_READY = 'DATA_READY'
PAUSE = 'PAUSE'
RESUME = 'RESUME'
STOP = 'STOP'
RUN_APPLICATION = 'RUN-APPLICATION'
LOAD_PIPELINE = 'LOAD-PIPELINE'
RUN_PIPELINE = 'RUN-PIPELINE'
SET_SETTING = 'SET-SETTING'
GET_SETTING = 'GET-SETTING'
PROGRESS = 'PROGRESS'
SET_SEQUENCE = 'SET-SEQUENCE'
SET_MODEL_STATE = 'SET-MODEL-STATE'
GET_MODEL_STATE = 'GET-MODEL-STATE'
GET_STATE = 'GET-STATE'
EXECUTE_CALL = 'EXECUTE-CALL'


class DispatcherBlockingAutoRemoveDeadEndPoints(DispatcherBlocking):
    """TODO: integrate this into esapy_dispatcher."""

    def _find_endpoints(self, sender='', signal=''):
        """TBW.

        :param sender:
        :param signal:
        :return:
        """
        endpoints = super()._find_endpoints(sender=sender, signal=signal)
        self._remove_dead_endpoints(endpoints, sender, signal)
        return super()._find_endpoints(sender=sender, signal=signal)

    def _remove_dead_endpoints(self, endpoints, sender='', signal=''):
        """TBW.

        :param endpoints:
        :param sender:
        :param signal:
        :return:
        """
        # automatically remove and dead end points
        if not endpoints:
            return
        dead_endpoints = []
        for endpoint_ref in endpoints:
            try:
                obj_ref = from_ref(endpoint_ref)  # type: t.Any
                if obj_ref is None:
                    dead_endpoints.append(endpoint_ref)
            except ReferenceError:
                dead_endpoints.append(endpoint_ref)

        for dead_endpoint in dead_endpoints:
            self._log.warning('Removing end-point: sender:%r, signal:%r, dead_endpoint: %r',
                              sender, signal, dead_endpoint)
            endpoints.remove(dead_endpoint)


dispatcher = DispatcherBlockingAutoRemoveDeadEndPoints()
# send_to_influxdb = dispatcher.emit(sender='*', signal=HK_SIGNAL)
progress = dispatcher.emit(sender='api', signal=PROGRESS)


class SequencerState:
    """The state of the sequencer."""

    error = -1
    idle = 0
    running = 1
    pause = 2
    completed = 3
    aborted = 4

    @staticmethod
    def to_string(state):
        """TBW.

        :param state:
        :return:
        """
        state_map = {
            SequencerState.error: 'error',
            SequencerState.idle: 'idle',
            SequencerState.running: 'running',
            SequencerState.pause: 'paused',
            SequencerState.completed: 'completed',
            SequencerState.aborted: 'aborted'
        }
        return state_map[state]