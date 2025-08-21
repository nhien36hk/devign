from ..utils.objects import stats
from tqdm import tqdm


class LoaderStep:
    def __init__(self, name, data_loader, device):
        self.name = name
        self.loader = data_loader
        self.size = len(data_loader)
        self.device = device

    def __call__(self, step):
        self.stats = stats.Stats(self.name)

        iterator = tqdm(self.loader, desc=self.name, leave=False)
        for i, batch in enumerate(iterator):
            batch.to(self.device)
            stat: stats.Stat = step(i, batch, batch.y)
            self.stats(stat)

        return self.stats
