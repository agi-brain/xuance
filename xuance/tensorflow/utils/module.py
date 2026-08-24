from abc import ABC
from typing import Dict, Iterator

import keras


class Module(keras.Model, ABC):

    def clone(
            self,
            *,
            copy_weights: bool = True,
            trainable: bool | None = None,
            name: str | None = None
    ):
        config = dict(self.get_config())

        if name is not None:
            config["name"] = name

        cloned = type(self).from_config(config)

        if copy_weights:
            if len(cloned.weights) == 0 and len(self.weights) > 0:
                build_config = self.get_build_config()
                if build_config:
                    cloned.build_from_config(build_config)

            if len(cloned.weights) != len(self.weights):
                raise RuntimeError(
                    f"Cannot copy weights from {type(self).__name__}: "
                    f"source has {len(self.weights)} weights, "
                    f"clone has {len(cloned.weights)} weights."
                )

            cloned.set_weights(self.get_weights())

        if trainable is not None:
            cloned.trainable = trainable

        return cloned


class ModuleList(keras.layers.Layer):
    def __init__(self, modules=None, **kwargs):
        super().__init__(**kwargs)
        self._modules = []

        if modules is not None:
            self.extend(modules)

    def __getitem__(self, index):
        return self._modules[index]

    def __setitem__(self, index, module):
        self._modules[index] = module

    def __delitem__(self, index):
        del self._modules[index]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def append(self, module):
        self._modules.append(module)
        return self

    def extend(self, modules):
        self._modules.extend(modules)
        return self

    def insert(self, index, module):
        self._modules.insert(index, module)

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            "ModuleList is only a container and has no forward computation."
        )


class ModuleDict(keras.layers.Layer):
    def __init__(self, modules: Dict[str, keras.layers.Layer] = None, **kwargs):
        super().__init__(**kwargs)

        self._modules = {}

        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> keras.layers.Layer:
        return self._modules[key]

    def __setitem__(self, key: str, module: keras.layers.Layer):
        if not isinstance(module, keras.layers.Layer):
            raise TypeError(
                f"module must be a keras.layers.Layer, "
                f"but got {type(module)}."
            )

        self._modules[key] = module

        # Register it as an attribute so Keras tracks the sublayer.
        setattr(self, f"_module_{key}", module)

    def __delitem__(self, key: str):
        del self._modules[key]
        delattr(self, f"_module_{key}")

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def get(self, key: str, default=None):
        return self._modules.get(key, default)

    def update(self, modules: Dict[str, keras.layers.Layer]):
        for key, module in modules.items():
            self[key] = module
