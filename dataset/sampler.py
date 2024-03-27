from collections import Counter

import torch
import math
import torch.distributed as dist

class DistributedWeightedSampler(torch.utils.data.Sampler):

    def __init__(self, weights, dataset, num_replicas=None, rank=None, replacement=True):
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.dataset = dataset
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ProgressiveSampler(torch.utils.data.Sampler):

    def __init__(self, sample_weight, num_samples, num_epochs, replacement=True,  interpolate='linear'):
        super().__init__()
        self.sample_weight = sample_weight
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.replacement = replacement
        self.interpolate = interpolate
        self.init_weight = torch.ones_like(sample_weight) / num_samples
        self.cur_epoch = 0

        if self.interpolate == 'linear':
            self.a = (self.sample_weight - self.init_weight) / self.num_epochs
        elif self.interpolate == 'quad':
            self.a = (self.sample_weight - self.init_weight) / (self.num_epochs**2)
        elif self.interpolate == 'inverse_quad':
            self.a = (self.sample_weight - self.init_weight) / math.sqrt(self.num_epochs)
        else:
            raise NotImplementedError

    def __get_current_weight__(self):
        if self.interpolate == 'linear':
            return self.a * self.cur_epoch + self.init_weight
        elif self.interpolate == 'quad':
            return self.a * (self.cur_epoch**2) + self.init_weight
        elif self.interpolate == 'inverse_quad':
            return self.a * math.sqrt(self.cur_epoch) + self.init_weight

    def __iter__(self):
        cur_weight = self.__get_current_weight__()
        self.cur_epoch += 1
        cur_index = torch.multinomial(cur_weight, num_samples=self.num_samples, replacement=self.replacement)
        return iter(cur_index.tolist())
    
    def __len__(self):
        return self.num_samples


def get_sampler(train_dataset, world_size, rank, args):
    if args.balance_data or args.binary_balance:
        if args.patch_lv:
            y_train = [int(train_dataset.labels[i]) for i in range(len(train_dataset))]
        else:
            y_train = [int(train_dataset.labels[i]) for i in train_dataset.train_idx]
        if args.binary_balance:
            if args.vindr:
                y_train = [0 if lb < 2 else 1 for lb in y_train]
            else:
                y_train = [0 if lb < 1 else 1 for lb in y_train]
        train_cnt = Counter(y_train)
        train_dist_list = {k: train_cnt[k] for k in train_cnt.keys()}
        class_weight = {k: 1./cnt for k, cnt in train_dist_list.items()}
        y_weight = torch.tensor([class_weight[label] for label in y_train])
        if args.progressive_sampler:
            train_sampler = ProgressiveSampler(y_weight, len(y_train), args.epochs, interpolate=args.sampling_interpolate)
        elif args.ddp:
            train_sampler = DistributedWeightedSampler(
                y_weight, 
                train_dataset, 
                num_replicas=world_size, 
                rank=rank
            )
        else:
            train_sampler = torch.utils.data.WeightedRandomSampler(y_weight, len(y_train))
    else:
        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas=world_size, 
                rank=rank
            )

    return train_sampler