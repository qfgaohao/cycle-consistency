

class Accumulator:
    def __init__(self):
        self.acc = {}
        self.counter = 0

    def update(self, updates: dict):
        for k, v in updates.items():
            self.acc[k] = self.acc.get(k, 0) + v
        self.counter += 1

    def mean(self):
        return {k: v/self.counter for k, v in self.acc.items()}

    def reset(self):
        self.acc = {}
        self.counter = 0

    def __str__(self):
        s = ""
        for k, v in self.acc.items():
            s += f"{k}: {v/self.counter:.4f}, "
        s = s[:-2]
        return s