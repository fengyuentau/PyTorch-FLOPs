from collections import OrderedDict

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

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

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    @property
    def flops(self):
        if self._flops != 0:
            self._flops = int(0)
        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                self._flops += module._flops
        return self._flops

    @property
    def name(self):
        return self.__class__.__name__