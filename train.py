
import torch
from torch.utils.data import DataLoader
from dataset import ESC50
from model import Net

dev = "cuda" if torch.cuda.is_available() else "cpu"
root = "ESC-50"

train = DataLoader(
    ESC50(root, [1, 2, 3, 4]),
    batch_size=32,
    shuffle=True
)

val = DataLoader(
    ESC50(root, [5]),
    batch_size=32
)

model = Net().to(dev)
opt = torch.optim.AdamW(
    model.parameters(),
    1e-3,
    weight_decay=1e-2
)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()

    for x, y in train:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    model.eval()
    ok = 0

    with torch.no_grad():
        for x, y in val:
            ok += (
                model(x.to(dev)).argmax(1).cpu() == y
            ).sum().item()

    print(epoch, ok / len(val.dataset))

torch.save(model.state_dict(), "esc50_cnn.pth")
print("Model saved to esc50_cnn.pth")
