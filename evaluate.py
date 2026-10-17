
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay

from dataset import ESC50
from model import Net

dev = "cuda" if torch.cuda.is_available() else "cpu"

val = DataLoader(
    ESC50("ESC-50", [5]),
    batch_size=32
)

model = Net().to(dev)
model.load_state_dict(
    torch.load("esc50_cnn.pth", map_location=dev)
)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, y in val:
        predictions = model(x.to(dev)).argmax(1).cpu()
        y_true.extend(y.tolist())
        y_pred.extend(predictions.tolist())

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    xticks_rotation=90
)

plt.tight_layout()
plt.show()
