
class NoneSchedule(object):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.constant_lr = lr
        self.step(0)

    def step(self, num_updates):
        self.lr = self.constant_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self):
        return self.get_lr()

class WarmupSchedule(NoneSchedule):
    def __init__(self, optimizer, lr, warmup_updates):
        self.optimizer = optimizer
        self.constant_lr = self.lr = lr
        self.warmup_updates = warmup_updates
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        self.lr = max(constant_lr * warmup, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        return self.lr