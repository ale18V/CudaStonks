import click
from sklearn.model_selection import train_test_split
from CudaStonks.constants import TEST_RATIO, epochs, learning_rate, regularization_strength
from CudaStonks.utils import plot
from CudaStonks.dataloader import loaddata, preprocess  # noqa: E402
from CudaStonks.constants import layers_size  # noqa: E402
import time


@click.group()
@click.option("--plot-path", type=str, help="Save plot to the given path.")
@click.pass_context
def cli(ctx, plot_path):
    ctx.ensure_object(dict)
    ctx.obj["PLOT_PATH"] = plot_path


@cli.command()
@click.option("--gpu/--cpu", is_flag=True, help="Run on gpu or cpu.")
@click.pass_context
def handcrafted(ctx, gpu: bool):
    from CudaStonks.handcrafted.activation import ReLU
    from CudaStonks.handcrafted.loss import MSELoss
    from CudaStonks.handcrafted.neuralnetwork import NeuralNetwork
    from CudaStonks.handcrafted.models import NeuralNetworkInterface
    from CudaStonks.handcrafted.optimizers import Adam

    # Load [features, label] for each stock
    for X, Y in loaddata():
        X, Y = preprocess(X, Y)

        trainX, testX, trainY, testY = train_test_split(
            X, Y, test_size=TEST_RATIO, shuffle=False)

        m, n = X.shape
        NN: NeuralNetworkInterface = NeuralNetwork
        if gpu:
            from CudaStonks.handcrafted.gpu.neuralnetwork import NeuralNetworkGPU
            NN = NeuralNetworkGPU

        opt = Adam(eta=learning_rate)
        model = NN(dim_features=n, dim_label=1, dim_hidden_layers=layers_size,
                   activation_fun=ReLU(),
                   loss_fun=MSELoss(), optimizer=opt)

        # Train the model and measure the time it takes
        print("Training model")
        start = time.time()
        model.train(trainX, trainY, lam=regularization_strength, epoch=epochs)
        end = time.time()
        print("Time elapsed:", end - start)

        predTestY = model.predict(testX)
        predTrainY = model.predict(trainX)
        plot(trainY, testY, predTrainY, predTestY, ctx.obj["PLOT_PATH"])


@cli.command()
@click.pass_context
def pytorch(ctx):
    import torch
    from torch import nn, optim
    from CudaStonks.pytorch.constants import device
    from CudaStonks.pytorch.model import StockPredictor

    for X, Y in loaddata():
        X, Y = preprocess(X, Y)

        trainX, testX, trainY, testY = train_test_split(
            X, Y, test_size=TEST_RATIO, shuffle=False)

        m, n = X.shape
        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float64)
        trainY = torch.from_numpy(trainY).to(device, dtype=torch.float64)
        testX = torch.from_numpy(testX).to(device, dtype=torch.float64)
        model = StockPredictor(n)

        # Move the model to GPU if available
        model.to(device)

        # Loss function (Mean Squared Error)
        criterion = nn.MSELoss(reduction='sum')

        optimizer = optim.Adam(model.parameters(), lr=0.0001,
                               weight_decay=1e-5)

        def train(model, criterion, optimizer, X, Y, epochs=2000):
            model.train()
            for epoch in range(epochs):
                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, Y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

        print("Training model")
        start = time.time()
        train(model, criterion, optimizer, trainX, trainY, epochs=epochs)
        end = time.time()
        print("Time elapsed:", end-start)
        model.eval()
        with torch.no_grad():
            predTestY = model(testX).cpu().numpy()
            predTrainY = model(trainX).cpu().numpy()

        plot(trainY.cpu(), testY, predTrainY, predTestY, ctx.obj["PLOT_PATH"])


cli()
