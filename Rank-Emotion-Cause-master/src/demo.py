import sys
import os
import warnings
import time
import numpy as np
import torch
from config import Config
from data_loader import build_inference_data
from main import inference_one_batch
from networks.rank_cp import Network
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import to_np, read_b, logistic, float_n, eval_func, write_b



def inference_one_epoch(configs, batches, model):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    for batch in batches:
        _, _, _, doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(configs, batch, model)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)

    return doc_id_all, doc_couples_all, doc_couples_pred_all

configs = Config()
model = Network(configs)

# Load pretrained model weights
model.load_state_dict(torch.load('model_params_8.pt'))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare test data loader
test_loader = build_inference_data(configs, fold_id=1, data_type='test')

# Iterate over test data and make predictions
all_predictions = []
for batch in test_loader:
    predictions = inference_one_batch(configs, batch, model)
    all_predictions.extend(predictions)

# Print or further process the predictions
print(all_predictions)
