import random
from torch.utils.data import Sampler

class BalancedSampler(Sampler):

    def __init__(self, dataset, ratios=None, samples_per_epoch=None):
        self.dataset = dataset

        # 每个 epoch 采多少样本
        self.num_samples = samples_per_epoch or len(dataset)

        # 后者为我的实验默认
        self.ratios = ratios or {
            0: 0.15,
            1: 0.15,
            2: 0.15,
            3: 0.15,
            4: 0.15,
            5: 0.15,
            6: 0.10
        }

        self.type_indices = dataset.type_indices

    def __iter__(self):
        indices = []

        for t, ratio in self.ratios.items():
            group = self.type_indices[t]
            n = int(self.num_samples * ratio)

            if len(group) == 0:
                continue

            if len(group) >= n:
                sampled = random.sample(group, n)
            else:
                sampled = random.choices(group, k=n)

            indices.extend(sampled)

        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples