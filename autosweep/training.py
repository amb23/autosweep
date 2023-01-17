
from tqdm import tqdm
import numpy as np
from autosweep.board import Cell, Result, create_board, dry_reveal, is_complete

import torch
from torch import nn
from torch.utils.data import DataLoader


def create_input(cells: list[Cell], n, m) -> np.ndarray:
    out = -1 * np.ones(len(cells))
    for i, cell in enumerate(cells):
        if cell.revealed:
            out[i] = cell.n_mines_adjacent
    return torch.from_numpy(out)


def labels_and_reveals(cells: list[Cell]) -> tuple[np.ndarray, dict[int, set[int]]]:
    """We label each cell on the board by the number of cells such a press would reveal"""
    out = np.zeros(len(cells))
    reveals: dict[set[int]] = {}
    for i, cell in enumerate(cells):
        if cell.loc in reveals:
            continue
        result, revealed, zeros = dry_reveal(i, cells)
        if result == Result.OKAY:
            out[i] = len(revealed)
        if result == Result.NO_OP:
            out[i] = -0.25
        if result == Result.INVALID:
            out[i] = -1
        if result == Result.GAME_OVER:
            out[i] = -1
        for j in zeros:
            reveals[j] = revealed
        for j in revealed - zeros:
            reveals[j] = {j}
    return out, reveals


def normalize(reveals):
    return np.where(reveals > 0, reveals / reveals.max(), reveals)


def create_batch_data(n, m, n_mines, n_boards):
    x = list()
    y = list()
    boards = list()
    for _ in tqdm(range(n_boards)):
        board = create_board(n, m, n_mines)
        labels, reveals = labels_and_reveals(board)
        n_plays = 0
        while not is_complete(board):
            x.append(create_input(board, n, m))
            y.append(torch.from_numpy(normalize(labels)))
            unrevealed_cells = [cell for cell in board if (not cell.revealed and not cell.is_mine)]
            # pick a random unrevealed cell and reveal it
            h = np.random.randint(len(unrevealed_cells))
            cell = unrevealed_cells[h]
            cell_reveals = reveals[cell.loc]
            for i in cell_reveals:
                board[i].revealed = True
                labels[i] = -0.25
            n_plays += 1
        boards.append((board, n_plays))
    return x, y, boards


def human_readable(data, n, m):
    # calculate the fixed width
    m_ = max(data)
    n_ = min(data)
    if m_ <= 0:
        fw = 1
    else:
        fw = 1 + int(np.log10(m_))
    if n_ < 0:
        fw = max(fw, 2 + int(np.log10(-n_)))
    for i in range(n):
        row = ""
        for j in range(m):
            if data[i +  j * n] == -2:
                row += " "*(fw - 1) + "- "
            elif data[i +  j * n] == -1:
                row += " "*(fw - 1) + "* "
            else:
                row += format(int(data[i+j*n]), f"{fw}d") + " "
        print(row)


def human_readable_2d(data, dtype=float):
    n, m = data.shape
    for i in range(n):
        row = ""
        for j in range(m):
            val = data[i, j]
            if dtype == int:
                row += f"{int(val):3d} "
            else:
                if val >= 0:
                    row += " "
                row += f"{val:.4f} "
        print(row)


def human_readable_floats(data, n, m):
    for i in range(n):
        row = ""
        for j in range(m):
            val = data[i + j *n]
            if val >= 0:
                row += " "
            row += f"{val:.4f} "
        print(row)


class Dataset:

    def __init__(self, n, m, n_mines, n_boards):
        self.n = n
        self.m = m
        self.n_mines = n_mines
        X, y, _ = create_batch_data(n, m, n_mines, n_boards)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]



class NeuralNetwork(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n * m, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(512, m, dtype=torch.float64)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train(dl, model, loss_fn, optimizer):
    size = len(dl.dataset)
    model.train()
    for batch, (X, y) in enumerate(dl):
        # error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:5f} [{current:>5d}/{size:>5d}]")


def test(dl, model, loss_fn):
    size = len(dl.dataset)
    n_batches = len(dl)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dl:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= n_batches
    print(f"Avg loss: {test_loss:>7f}")


if __name__ == "__main__":
    n = 8
    m = 8
    n_mines = 10

    model = NeuralNetwork(n, m)
    print(model)
    batch_size = 100
    test_dl = DataLoader(Dataset(n, m, n_mines, 1000))
    train_dl = DataLoader(Dataset(n, m, n_mines, 100))

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print (f"Epoch {t+1}")
        print ("-----------------------------------------")
        train(train_dl, model, loss_fn, optimizer)
        test(test_dl, model, loss_fn)


    torch.save(model.state_dict(), "model.pth")
    print ("Model saved to `model.pth'")

