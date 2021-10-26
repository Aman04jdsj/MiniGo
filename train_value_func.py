import csv
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from random import shuffle


LOG_DIR = "results"
ITERATIONS = 100
# Hyper parameters
INPUT_NEURONS = 25
HIDDEN_NEURONS = 256
OUTPUT_NEURONS = 1
BATCH_SIZE = 8


class GoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input to hidden layer
        self.hidden1 = nn.Linear(INPUT_NEURONS, HIDDEN_NEURONS)
        self.hidden2 = nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
        self.output = nn.Linear(HIDDEN_NEURONS, OUTPUT_NEURONS)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    with open("black.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    # with open("white.csv", "r") as f:
    #     reader = csv.reader(f)
    #     data += list(reader)
    X = np.array([[int(j) for j in list(i[0].strip('][').split(', '))] for i in data])
    Y = np.array([float(i[1]) for i in data])
    model = GoModel()
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    number_of_batches = X.shape[0]//BATCH_SIZE
    idx_list = [i for i in range(X.shape[0])]
    for epoch in range(ITERATIONS):
        # shuffle the data
        shuffle(idx_list)
        X = X[idx_list]
        Y = Y[idx_list]
        epoch_loss = 0
        for batch in range(number_of_batches):
            optimizer.zero_grad()
            X_batch = X[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            Y_batch = Y[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            X_tensor = torch.Tensor(X_batch)
            X_tensor = torch.reshape(X_tensor, (BATCH_SIZE, INPUT_NEURONS))
            Y_tensor = torch.FloatTensor(Y_batch)
            Y_tensor = torch.reshape(Y_tensor, (BATCH_SIZE, OUTPUT_NEURONS))
            Y_pred = model(X_tensor)
            loss = criterion(Y_pred, Y_tensor)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", epoch_loss/number_of_batches, epoch)
        X_tensor = torch.Tensor(X)
        X_tensor = torch.reshape(X_tensor, (X.shape[0], INPUT_NEURONS))
        model.eval()
        Y_pred = model(X_tensor).detach().numpy()
        loss = np.linalg.norm(Y_pred - Y)
        writer.add_scalar("Loss/test", loss, epoch)
        torch.save(model.state_dict(), LOG_DIR+"/model_"+str(epoch))
    writer.flush()

    # Loading model
    testing_model = GoModel()
    testing_model.load_state_dict(torch.load(LOG_DIR+"/model_9"))
    X_tensor = torch.Tensor(X)
    X_tensor = torch.reshape(X_tensor, (X.shape[0], INPUT_NEURONS))
    testing_model.eval()
    Y_pred = testing_model(X_tensor).detach().numpy()
    loss = np.linalg.norm(Y_pred - Y)
    print(f"The final loss is {loss}")
