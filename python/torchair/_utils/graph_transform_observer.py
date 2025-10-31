from torch.fx.graph_module import GraphModule


class GraphTransformObserver:
    '''
    Custom FX Pass observer to print debug information before and after FX Pass execution.
    '''
    __pass_count = 0

    def __init__(self, gm: GraphModule, pass_name: str, phase: str):
        self._gm = gm
        self._pass_name = pass_name
        self._phase = phase

        from torchair import logger
        self._logger = logger
        GraphTransformObserver.__pass_count += 1
    
    def __enter__(self):
        self._logger.debug('PASS_%s [%s] before [%s] execution, graph is %s', GraphTransformObserver.__pass_count, self._phase, self._pass_name, self._gm.graph)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._logger.debug('PASS_%s [%s] after [%s] execution, graph is %s', GraphTransformObserver.__pass_count, self._phase, self._pass_name, self._gm.graph)
            return True
        else:
            raise exc_val.with_traceback(exc_tb)
    
    @classmethod
    def get_current_pass_count(cls):
        return cls.__pass_count
    
    def apply_gm_pass(self, pass_fn, example_inputs, config):
        pass_fn(self._gm, example_inputs, config)
