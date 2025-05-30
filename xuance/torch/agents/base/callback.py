from abc import ABC


class BaseCallback(ABC):
    def on_train_start(self, *args, **kwargs): pass

    def on_train_log(self, *args, **kwargs): pass

    def on_train_step(self, *args, **kwargs): pass

    def on_train_end(self, *args, **kwargs): pass

    def on_update_start(self, *args, **kwargs): pass

    def on_update_end(self, *args, **kwargs): pass
