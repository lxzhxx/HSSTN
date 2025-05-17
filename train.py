import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
from tqdm import trange
import shutil
from eval import eval_od_prediction
from Hierarchical.hsstn import HSSTNet
from utils import EarlyStopMonitor
from utils import get_od_data
import torch.nn as nn
import torch.nn.functional as F

config = {
    "liuyang": {
        "train_day": 18,
        "val_day": 6,
        "test_day": 6,
    }
}

parser = argparse.ArgumentParser('train')
parser.add_argument('-d', '--data', type=str ,default='liuyang')
parser.add_argument('--seed', type=int, default=4396, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:0", help='Idx for the gpu to use: cpu, cuda:0, etc.')

parser.add_argument('--loss', type=str, default="odloss", help='Loss function')
parser.add_argument('--message_dim', type=int, default=256, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=256, help='Dimensions of the memory for '
                                                               'each node')
parser.add_argument('--lambs', type=float, nargs="+", default=[1], help='Lamb of different time scales')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

NUM_EPOCH = args.n_epoch
device = args.device
DATA = args.data
LEARNING_RATE = args.lr
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

input_len = 3600
output_len = 3600
day_cycle = 86400
day_start = 0
day_end = 86400

Path("./model/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./model/{args.data}-{args.suffix}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.data}-{args.suffix}-{epoch}.pth'
results_path = f"results/{args.data}_{args.suffix}.pkl"
Path("results/").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f"log/{str(time.time())}_{args.suffix}.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix, back_points = get_od_data(config[DATA])
model = HSSTNet(device=device, n_nodes=n_nodes, node_features=node_features,
             message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
             output=output_len, lambs=args.lambs)

class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1).float()
        loss_mask = ((predict - truth) ** 2) * (1 - mask)
        masked_loss = (mask * (self.relu(predict) - truth) ** 2)
        loss = torch.mean(loss_mask + masked_loss)
        return loss

class ODKL_Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("interval_weights", torch.tensor([0.1, 1.0, 2.0, 3.0]))

    def _get_distribution(self, matrix):
        if matrix.dim() == 1:
            matrix = matrix.view(256, 256)
        out_degree = matrix.sum(dim=1) + self.epsilon
        in_degree = matrix.sum(dim=0) + self.epsilon
        combined = torch.stack([out_degree, in_degree], dim=1)
        return F.softmax(combined, dim=1)

    def forward(self, predict, truth):
        masks = [
            (truth == 0),
            (truth > 0) & (truth < 2),
            (truth >= 2) & (truth < 4),
            (truth >= 4)
        ]
        interval_losses = []
        for i, mask in enumerate(masks):
            diff = (predict - truth) * mask.float()
            loss = (diff ** 2) * self.interval_weights[i]
            interval_losses.append(loss)
        od_loss = torch.mean(torch.stack(interval_losses).sum(dim=0))
        num = truth.shape[0]
        kl_loss = 0.0
        for b in range(num):
            p_b = self._get_distribution(truth[b])
            p_hat_b = self._get_distribution(predict[b])
            kl_div = (p_b * (torch.log(p_b + self.epsilon) -
                             torch.log(p_hat_b + self.epsilon))).sum()
            kl_loss += kl_div
        return od_loss + 0.6 * (kl_loss / num)

if args.loss == "odloss":
    criterion = ODKL_Loss()

model = model.to(device)

val_mses = []
epoch_times = []
total_epoch_times = []
train_losses = []
if args.best == "":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
    num_batch = (val_time - input_len) // input_len
    for epoch in range(NUM_EPOCH):
        print("================================Epoch: %d================================" % epoch)
        start_epoch = time.time()
        logger.info('start {} epoch'.format(epoch))
        m_loss = []

        model.init_memory()
        model = model.train()
        batch_range = trange(num_batch)
        for j in batch_range:

            begin_time = j * input_len
            now_time = j * input_len + input_len
            if now_time % day_cycle < day_start or now_time % day_cycle > day_end:
                continue

            head, tail = back_points[begin_time // input_len], back_points[
                now_time // input_len]  # [head,tail1) nowtime [tail1,tail2) nowtime+τ
            if head == tail:
                continue

            time_of_matrix = now_time % day_cycle // input_len
            weekday_of_matrix = now_time // day_cycle % 7
            time_of_matrix2 = (now_time + input_len) % day_cycle // input_len
            weekday_of_matrix2 = (now_time + input_len) // day_cycle % 7

            optimizer.zero_grad()
            sources_batch, destinations_batch = full_data.sources[head:tail], full_data.destinations[head:tail]
            edge_idxs_batch = full_data.edge_idxs[head:tail]
            timestamps_batch_torch = torch.Tensor(full_data.timestamps[head:tail]).to(device)
            time_diffs_batch_torch = torch.Tensor(full_data.timestamps[head:tail] - now_time).to(device)

            if now_time % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True

            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                predict_od=predict_od)

            if predict_od:
                od_matrix_real = od_matrix[now_time // input_len]
                loss = criterion(od_matrix_predicted, torch.FloatTensor(od_matrix_real).to(device))
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

            batch_range.set_description(f"train_loss: {m_loss[-1]};")

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        print("================================Val================================")
        val_mse, val_rmse, val_mae, val_pcc, val_mape, _, _ = eval_od_prediction(model=model,
                                                                                  data=full_data,
                                                                                  od_matrix=od_matrix,
                                                                                  back_points=back_points,
                                                                                  st=val_time,
                                                                                  ed=test_time,
                                                                                  device=device,
                                                                                  config=config[DATA])

        val_mses.append(val_mse)
        train_losses.append(np.mean(m_loss))
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        pickle.dump({
            "val_mses": val_mses,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
        logger.info(
            f'Epoch val metric: mae, mse, rmse, pcc, mape, {val_mae}, {val_mse}, {np.sqrt(val_mse)}, {val_pcc}, {val_mape}')
        # Early stopping
        ifstop, ifimprove = early_stopper.early_stop_check(val_mse)
        if ifstop:
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(
                {"statedict": model.state_dict(), "memory": model.backup_memory()},
                get_checkpoint_path(epoch))
    logger.info('Saving DyOD model')
    shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
    logger.info('DyOD model saved')
    best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
else:
    best_model_param = torch.load(args.best)

model.load_state_dict(best_model_param["statedict"])
model.restore_memory(best_model_param["memory"])

print("================================Test================================")
test_mse, test_rmse, test_mae, test_pcc, test_mape, prediction, label = eval_od_prediction(model=model,
                                                                                            data=full_data,
                                                                                            od_matrix=od_matrix,
                                                                                            back_points=back_points,
                                                                                            st=test_time,
                                                                                            ed=all_time,
                                                                                            device=device,
                                                                                            config=config[DATA])

logger.info(
    'Test statistics:-- mae: {}, mse: {}, rmse: {}, pcc: {}, mape:{}'.format(test_mae, test_mse, test_rmse, test_pcc,
                                                                              test_mape))
pickle.dump({
    "val_mses": val_mses,
    "test_mse": test_mse,
    "test_rmse": np.sqrt(test_mse),
    "test_mae": test_mae,
    "test_pcc": test_pcc,
    "test_mape": test_mape,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times,
    "prediction": prediction,
    "label": label
}, open(results_path, "wb"))
