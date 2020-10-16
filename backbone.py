import torch.nn as nn

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_features, hidden_size=64, embedding_size=256):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.projection = nn.Linear(embedding_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.projection(features)
        return logits


class ProjectionHead(nn.Module):

    def __init__(self, in_embedding_size: int, projection_size: int):
        super().__init__()
        self.hidden = nn.Linear(in_embedding_size, projection_size, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(projection_size, projection_size, bias=False)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, dim=1)
        return x

class SimCLR(nn.Module):
    EMBEDDING_SIZE = 128
    PROJECTION_SIZE = 64
    def __init__(self):
        super().__init__()
        self.encoder = ConvolutionalNeuralNetwork(3, 128)
        self.projection_head = ProjectionHead(self.EMBEDDING_SIZE, self.PROJECTION_SIZE)

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return embedding, projection
