from abc import ABC


class Callback(ABC):
    def on_train_start(self, *args, **kwargs): pass

    def on_train_step(self, *args, **kwargs): pass

    def on_train_end(self, *args, **kwargs): pass

    def on_update_start(self, *args, **kwargs): pass

    def on_update_end(self, *args, **kwargs): pass
