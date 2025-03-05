from matplotlib import pyplot as plt


def plot(trainY, testY, predTrainY, predTestY, path: str):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(range(len(testY)), testY, c='r', label='Actual')
    axes[1].plot(range(len(trainY)), trainY, c='r', label='Actual')

    axes[0].plot(range(len(predTestY)), predTestY,
                 c='b', label='Predicted')
    axes[1].plot(range(len(predTrainY)), predTrainY,
                 c='b', label='Predicted')

    axes[0].set_title('Testing Set')
    axes[1].set_title('Training Set')
    for ax in axes:
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        ax.legend()
    plt.show()
    if path:
        fig.savefig(path)
