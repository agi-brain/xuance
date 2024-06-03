
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl import AddSelfLoop
from dgl.data import CoraGraphDataset

transform = (
    AddSelfLoop()
)
data = CoraGraphDataset(transform=transform)
g = data[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = g.int().to(device)
features = g.ndata["feat"]
labels = g.ndata["label"]
masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
in_size = features.shape[1]
out_size = data.num_classes
print('*' * 100)
print(g)
print('特征维度', features.shape)
print('输出维度', len(torch.unique(labels)))
print('边的条数', len(g.edges()[0]))
print('边', g.edges())


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, x):
        h = self.conv1(g, x).flatten(1)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.conv2(g, h).mean(1)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)

        logits = logits[mask]
        labels = labels[mask]
        # probabilities = F.softmax(logits, dim=1)
        # print(probabilities)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, epoches):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(epoches):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


model = GAT(in_size, 3, out_size, num_heads=10).to(device)

# model training
print("Training...")
epoches = 50
train(g, features, labels, masks, model, epoches)

# test the model
print("Testing...")
acc = evaluate(g, features, labels, masks[2], model)
print("Test accuracy {:.4f}".format(acc))