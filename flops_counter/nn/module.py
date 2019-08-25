from collections import OrderedDict

class Module(object):
    def __init__(self):
        # super(Module, self).__init__()

        # self._forward_hooks = OrderedDict()
        # self._forward_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
        self._flops = int(0)

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        return result

    def __setattr__(self, name, value):
        def remove_from(dicts):
            for d in dicts:
                if name in d:
                    print('remove {:s}.'.format(name))
                    del d[name]

        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    'cannot assign module before Module.__init__() call')
            remove_from(self.__dict__)
            modules[name] = value
        object.__setattr__(self, name, value)

    @property
    def flops(self):
        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                self._flops += module._flops
        return self._flops