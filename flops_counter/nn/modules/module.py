import flops_counter

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
        self._input = list()
        self._output = list()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        self._calc_flops(input[0], result)
        self._input = input[0]
        self._output = result
        return result

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError('{} is not a Module subclass'.format(
                flops_counter.typename(module)))
        elif not isinstance(name, flops_counter.string_classes):
            raise TypeError('module name should be a string. Got {}'.format(
                flops_counter.typename(name)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError('attribute \'{}\' already exists'.format(name))
        elif '.' in name:
            raise KeyError('module name can\'t contain "."')
        elif name == '':
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    'cannot assign module before Module.__init__() call')
            remove_from(self.__dict__)
            modules[name] = value
        else:
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

        if self._flops != 0:
            main_str += ', FLOPs = {:,d}, input = {:s}, output = {:s}'.format(self._flops, str(self._input), str(self._output))
        return main_str

    def _calc_flops(self, x, y):
        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                self._flops += module._flops

    def set_flops_zero(self):
        self._flops = 0
        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                module.set_flops_zero()
            else:
                module._flops = 0

    @property
    def flops(self):
        return self._flops