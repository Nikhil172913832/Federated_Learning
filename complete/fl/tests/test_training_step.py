import torch

from fl.models import SimpleCNN
from fl.task import train


def test_training_step_runs_one_batch():
    model = SimpleCNN()
    x = torch.randn(4, 1, 28, 28)
    y = torch.zeros(4, dtype=torch.long)
    ds = [{"image": x[i], "label": y[i]} for i in range(4)]

    class DL:
        def __iter__(self):
            return iter(ds)

        def __len__(self):
            return 1

        @property
        def dataset(self):
            return ds

    device = torch.device("cpu")
    loss = train(model, DL(), epochs=1, lr=0.001, device=device)
    assert isinstance(loss, float)
