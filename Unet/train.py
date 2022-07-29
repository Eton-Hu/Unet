from model.unet_model import UNet
from utils.dataset import CustomDataset
from torch import optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    # Load traning set
    isbi_dataset = CustomDataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True,
                                               )
    # Number of batches per epoch
    nbperepoch = len(train_loader)
    # Define optimizer
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # Define loss calculator
    criterion = nn.BCEWithLogitsLoss()
    # best_loss recorder
    best_loss = float('inf')
    losses = []
    counters = []

    # Init loss display panel
    # Enable interactive mode.
    plt.ion()
    # Create a figure and a set of subplots.
    figure, ax = plt.subplots()
    # return AxesImage object for using.
    lines, = ax.plot([], [])
    ax.set_autoscaley_on(True)
    # ax.set_xlim(min_x, max_x)
    ax.grid()

    for epoch in range(epochs):
        net.train()
        print(f"training epoch: {epoch}")
        bi = 1
        for image, label in train_loader: 
            training_round = nbperepoch * epoch + bi
            bi += 1
            optimizer.zero_grad()
            # copy data to device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # output predict
            pred = net(image)
            # calculate loss
            loss = criterion(pred, label)
            losses.append(loss.item())
            counters.append(training_round)
            print(f'In training round {training_round} Loss/train {loss.item()}')
            # save the best model
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # update weight
            loss.backward()
            optimizer.step()

            # Update display
            lines.set_ydata(losses)
            lines.set_xdata(counters)
            #Need both of these in order to rescale
            ax.relim()
            ax.autoscale_view()
            # draw and flush the figure .
            figure.canvas.draw()
            figure.canvas.flush_events()
    print(f"best loss:{best_loss}")
    while 1:
        pass


if __name__ == "__main__":
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    import os
    # Start train
    dir = r'.\data'
    dir = os.path.abspath(dir)
    train_net(net, device, dir)