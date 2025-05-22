from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def eval_od_prediction(model, data, od_matrix, back_points, st, ed, device, config):
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    label, prediction = [], []

    with torch.no_grad():
        model = model.eval()
        num_test_batch = (ed - st - output_len) // input_len

        for j in tqdm(range(num_test_batch)):
            begin_time = j * input_len + st
            now_time = j * input_len + input_len + st
            if now_time % day_cycle < day_start or now_time % day_cycle > day_end:
                continue
            head, tail = back_points[begin_time // input_len], back_points[
                now_time // input_len]

            if head == tail:
                continue

            if now_time % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True

            sources_batch, destinations_batch = data.sources[head:tail], data.destinations[head:tail]
            timestamps_batch_torch = torch.Tensor(data.timestamps[head:tail]).to(device)
            time_diffs_batch_torch = torch.Tensor(data.timestamps[head:tail] - now_time).to(device)

            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                predict_od=predict_od)

            if predict_od:
                prediction.append(od_matrix_predicted.cpu().numpy())
                od_matrix_real = od_matrix[now_time // input_len]
                label.append(od_matrix_real)

        stacked_prediction = np.stack(prediction)
        stacked_prediction[stacked_prediction < 0] = 0
        stacked_label = np.stack(label)

        od_pred_flat = stacked_prediction.reshape(-1)
        od_label_flat = stacked_label.reshape(-1)

        mse = mean_squared_error(od_label_flat, od_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(od_label_flat, od_pred_flat)
        pcc = np.corrcoef(od_pred_flat, od_label_flat)[0][1]
        mape = np.mean(np.abs((od_label_flat - od_pred_flat) / (od_label_flat + 1e-5))) * 100

        inflow_pred = stacked_prediction.sum(axis=1)
        outflow_pred = stacked_prediction.sum(axis=2)
        inflow_label = stacked_label.sum(axis=1)
        outflow_label = stacked_label.sum(axis=2)

        io_pred = np.concatenate([inflow_pred.reshape(-1), outflow_pred.reshape(-1)])
        io_label = np.concatenate([inflow_label.reshape(-1), outflow_label.reshape(-1)])

        mse_io = mean_squared_error(io_label, io_pred)
        rmse_io = np.sqrt(mse_io)
        mape_io = np.mean(np.abs((io_label - io_pred) / (io_label + 1e-5))) * 100

        print("\nOD Matrix Metrics:")
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}, MAPE: {mape:.4f}%")

        print("\nIO Flow Metrics (in+out combined):")
        print(f"MSE: {mse_io:.4f}, RMSE: {rmse_io:.4f}, MAPE: {mape_io:.4f}%")

        output_dir_data = "./results/liuyang/3600/"
        os.makedirs(output_dir_data, exist_ok=True)

    return mse, rmse, mae, pcc, mape, stacked_prediction, stacked_label
