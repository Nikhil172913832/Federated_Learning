"""fl: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
from fl.config import load_run_config
from fl.personalization import fedprox_loss
from fl.dp import attach_dp_if_enabled
from fl.partitioning import build_partitioner


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
    
    Note: This is the legacy model class. For new implementations, 
    use fl.models.SimpleCNN instead for better modularity.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)  # batch_size=1, 1 channel, 28x28 image
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            flattened_size = dummy.numel()

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

_cfg = load_run_config()
_pre = _cfg.get("preprocess", {})
_resize = _pre.get("resize")
_mean = _pre.get("normalize_mean", [0.5])
_std = _pre.get("normalize_std", [0.5])

_tfms = []
if _resize:
    _tfms.append(Resize(size=_resize))
_tfms.append(ToTensor())
_tfms.append(Normalize(tuple(_mean), tuple(_std)))

pytorch_transforms = Compose(_tfms)

# Optional augmentation pipeline (applied in dataset mapping if enabled)
_aug_cfg = _pre.get("augmentation", {"enabled": False, "params": {}})
_aug_enabled = bool(_aug_cfg.get("enabled", False))
_aug_params = _aug_cfg.get("params", {})

_aug_tfms = []
if _aug_enabled:
    if _aug_params.get("hflip", True):
        _aug_tfms.append(RandomHorizontalFlip(p=0.5))
    rot_deg = float(_aug_params.get("rotation_degrees", 0.0))
    if rot_deg > 0:
        _aug_tfms.append(RandomRotation(degrees=rot_deg))
    if _aug_params.get("color_jitter", False):
        _aug_tfms.append(ColorJitter(
            brightness=_aug_params.get("brightness", 0.0),
            contrast=_aug_params.get("contrast", 0.0),
            saturation=_aug_params.get("saturation", 0.0),
            hue=_aug_params.get("hue", 0.0),
        ))
_augmentations = Compose(_aug_tfms) if _aug_tfms else None



def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    imgs = batch["image"]
    if _augmentations is not None:
        imgs = [_augmentations(img) for img in imgs]
    batch["image"] = [pytorch_transforms(img) for img in imgs]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data.
    
    Args:
        partition_id: ID of the partition to load (must be in [0, num_partitions))
        num_partitions: Total number of partitions (must be > 0)
        
    Returns:
        Tuple of (trainloader, testloader)
        
    Raises:
        ValueError: If partition_id or num_partitions are invalid
    """
    # Input validation
    if num_partitions <= 0:
        raise ValueError(
            f"num_partitions must be positive, got {num_partitions}"
        )
    
    if not (0 <= partition_id < num_partitions):
        raise ValueError(
            f"partition_id must be in [0, {num_partitions}), got {partition_id}"
        )
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        data_cfg = _cfg.get("data", {})
        dataset = data_cfg.get("dataset", "albertvillanova/medmnist-v2")
        subset = data_cfg.get("subset", "pneumoniamnist")
        non_iid_cfg = data_cfg.get("non_iid")
        partitioner = build_partitioner(num_partitions=num_partitions, cfg=non_iid_cfg)
        fds = FederatedDataset(
            dataset=dataset,
            subset=subset,
            trust_remote_code=True,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    batch_size = int(_cfg.get("data", {}).get("batch_size", 32))
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device, global_state_dict=None):
    """Train the model on the training set.
    
    Args:
        net: Neural network model
        trainloader: DataLoader for training data
        epochs: Number of local epochs (must be >= 1)
        lr: Learning rate (must be in (0, 1])
        device: Device to train on
        global_state_dict: Optional global model parameters for FedProx
        
    Returns:
        Average training loss
        
    Raises:
        ValueError: If epochs or lr are invalid
    """
    # Input validation
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    
    if not (0 < lr <= 1.0):
        raise ValueError(f"lr must be in (0, 1], got {lr}")
    
    if len(trainloader.dataset) == 0:
        raise ValueError("trainloader dataset is empty")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Attach DP if enabled
    dp_cfg = _cfg.get("privacy", {}).get("dp_sgd", {})
    dp_engine = None
    if dp_cfg.get("enabled", False):
        result = attach_dp_if_enabled(net, optimizer, trainloader, device, dp_cfg)
        dp_engine, net, optimizer, trainloader = result  # type: ignore
    net.train()
    running_loss = 0.0
    mu = float(_cfg.get("personalization", {}).get("fedprox_mu", 0.0))
    use_fedprox = _cfg.get("personalization", {}).get("method", "none") == "fedprox" and mu > 0
    global_params = None
    if use_fedprox and global_state_dict is not None:
        global_params = [p.detach().clone() for p in global_state_dict.values()]

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if use_fedprox and global_params is not None:
                loss = loss + fedprox_loss(net, global_params, mu)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set.
    
    Args:
        net: Neural network model
        testloader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        Tuple of (loss, accuracy)
        
    Raises:
        ValueError: If testloader is empty
    """
    # Input validation
    if len(testloader.dataset) == 0:
        raise ValueError("testloader dataset is empty")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
