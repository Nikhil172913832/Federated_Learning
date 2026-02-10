"""fl: A Flower / PyTorch app."""

import torch
from fl.config import load_run_config
from fl.personalization import fedprox_loss
from fl.dp import attach_dp_if_enabled
from fl.data_loader import FederatedDataLoader


def load_data(partition_id: int, num_partitions: int):
    """Load partition data.

    Args:
        partition_id: ID of the partition to load (must be in [0, num_partitions))
        num_partitions: Total number of partitions (must be > 0)

    Returns:
        Tuple of (trainloader, testloader)

    Raises:
        ValueError: If partition_id or num_partitions are invalid
    """
    config = load_run_config()
    data_loader = FederatedDataLoader.get_instance(config)
    return data_loader.load_partition(partition_id, num_partitions)


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
    config = load_run_config()
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    dp_cfg = config.get("privacy", {}).get("dp_sgd", {})
    dp_engine = None
    if dp_cfg.get("enabled", False):
        result = attach_dp_if_enabled(net, optimizer, trainloader, device, dp_cfg)
        dp_engine, net, optimizer, trainloader = result
    net.train()
    running_loss = 0.0
    mu = float(config.get("personalization", {}).get("fedprox_mu", 0.0))
    use_fedprox = (
        config.get("personalization", {}).get("method", "none") == "fedprox" and mu > 0
    )
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
