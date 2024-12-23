import stageone
import stagetwo
import torch.nn as nn
import torch.optim as optim

def train_model(net,frame_array):
    net = stagetwo.selecsls.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    number_of_frames = frame_array.size
    number_of_3dsamples =

    for epoch in range(2):
        net.train()
        for frames in frame_array:
