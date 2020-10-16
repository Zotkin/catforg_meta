import torch
import torch.nn as nn

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # TODO this might potentially lead to problems
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def get_correlated_samples_mask(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):


        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        mask = self.get_correlated_samples_mask(batch_size)

        p1 = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )
        negative_samples = sim[mask].reshape(batch_size * 2, -1)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # TODO According to documentation https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html,
        # if you inherit from LightningModule your class should have attribute device
        # In practice, such attribute is absent
        labels = torch.zeros(batch_size * 2).to(self.device).long()

        loss = self.criterion(logits, labels)
        loss /= 2 * batch_size
        return loss