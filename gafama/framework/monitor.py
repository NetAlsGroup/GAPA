class State:
    def __init__(self, **kwargs):
        self.fitness_list = None
        self.__dict__.update(kwargs)


class Monitor:
    def __init__(self, state: State, side):
        self.state = state
        self.side = side

    @property
    def best_fitness(self):
        if self.side == "max":
            return max(self.state.fitness_list)
        elif self.side == "min":
            return min(self.state.fitness_list)
        else:
            raise ValueError(f"No such side. Please choose 'max' or 'min'.")
