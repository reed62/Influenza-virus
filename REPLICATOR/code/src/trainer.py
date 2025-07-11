from torch.utils.data.dataloader import DataLoader
import torch
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class TrainerConfig:
    max_epochs = 30
    batch_size = 64
    learning_rate = 3e-4
    num_workers = 1
    optimizer = "adam"
    early_stop = True
    patience = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_set, test_set, config):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.config = config
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def train(self):
        model, config = self.model, self.config
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

        def run_epoch(dataset):
            is_train = dataset == "train"
            model.train(is_train)
            data = self.train_set if is_train else self.test_set
            loader = DataLoader(
                data, batch_size=config.batch_size, num_workers=config.num_workers
            )
            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, batch in pbar:
                try:
                    batch = [item.to(self.device) for item in batch]

                    with torch.set_grad_enabled(is_train):
                        out, loss = model(*batch)
                    if torch.isnan(loss).any():
                        print(f"❌ NaN in loss at iter {it}. Skipping.")
                        continue

                    loss = loss.mean()
                    losses.append(loss.item())

                    if is_train:
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}.")

                except Exception as e:
                    print(f"❌ Error during forward/backward pass at iter {it}: {e}")
                    print(f"    Batch shapes: {[b.shape for b in batch]}")
                    continue

            if not is_train:
                if len(losses) == 0:
                    print("❌ No valid losses collected during evaluation!")
                    return float('nan')
                test_loss = np.mean(losses)
                print("Test loss: %.5f" % (test_loss))
                return test_loss

        last_loss = 100
        patience = config.patience
        trigger_times = 0
        for epoch in range(config.max_epochs):
            run_epoch("train")
            if self.test_set is not None:
                test_loss = run_epoch("test")
                #scheduler.step(test_loss)
                if config.early_stop:
                    if test_loss > last_loss:
                        trigger_times += 1
                        if trigger_times >= patience:
                            print("\nEarly stopping!\n")
                            break
                    else:
                        trigger_times = 0
                    last_loss = test_loss

    def evaluate(self, method="mse"):
        self.model.eval()
        data = self.test_set
        loader = DataLoader(
            data, batch_size=self.config.batch_size, num_workers=self.config.num_workers
        )
        losses = []
        for it, batch in enumerate(loader):
            batch = [item.to(self.device) for item in batch]
            with torch.set_grad_enabled(False):
                out, loss = self.model(*batch)
                if method == "mse":
                    loss = loss.mean()
                    losses.append(loss.item())
                elif method == "pearsonr":
                    all_out = []
                    all_y_true = []
                    for it, batch in enumerate(loader):
                        batch = [item.to(self.device) for item in batch]
                        with torch.no_grad():
                            out, _ = self.model(*batch)
                        all_out.append(out.cpu().numpy())
                        all_y_true.append(batch[1].cpu().numpy())
                    all_out = np.concatenate(all_out).flatten()
                    all_y_true = np.concatenate(all_y_true).flatten()
                    r, _ = pearsonr(all_out, all_y_true)
                    return r

                elif method == "r2":
                    out = out.cpu()
                    y_true = batch[1].cpu()
                    losses.append(r2_score(out, y_true))
        return np.mean(losses)
