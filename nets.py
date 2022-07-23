import torch


class Generator(torch.nn.Module):
    def __init__(self, in_features, out_features, n_layers):
        exponent = int(torch.log2(torch.tensor(in_features)).item())
        layers = list()
        n_features = in_features
        for i in range(1, n_layers):
            layers.append(torch.nn.Linear(in_features=n_features, out_features=2**(i + exponent)))
            layers.append(torch.nn.ReLU())
            n_features = 2**(i + exponent)
        layers.append(torch.nn.Linear(in_features=n_features, out_features=out_features))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_features, out_features, n_layers):
        exponent = int(torch.log2(torch.tensor(in_features)).item())
        layers = list()
        n_features = in_features
        for i in range(n_layers):
            layers.append(torch.nn.Linear(in_features=n_features, out_features=2 ** (exponent - i)))
            layers.append(torch.nn.ReLU())
            n_features = 2 ** (exponent - i)
        layers.append(torch.nn.Linear(in_features=n_features, out_features=out_features))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
