import numpy as np
from NanoNet import Network, ADAM, CategorialCrossEntropy, Sigmoid, SoftMax, DataLoader, ReLu, load_from_file
from NanoNet.data.examples import MNIST_DataSet_PKL


def main():
    # Note: No path needed now, it's automatic!
    training_data = MNIST_DataSet_PKL(type='train')
    validation_data = MNIST_DataSet_PKL(type='validation')

    training_loader = DataLoader(training_data, batch_size=23, shuffle=True, drop_last=False)

    net = Network([784, 100, 100, 50, 10], [ReLu(), ReLu(), ReLu(), SoftMax()], dropout_rate=0.2)
    cost_function = CategorialCrossEntropy(net, False, True, lambd=0.001)

    from tqdm import tqdm # Import this to use tqdm.write

    def epoch_callback(epoch):
        # Performance calculation
        results = [(np.argmax(net.feedforward(x)), np.argmax(y))
                            for (x, y) in validation_data]
        acc = sum(int(x == y) for (x, y) in results) / len(validation_data)
        
        # Optional: If you still want a clean line in the console history
        tqdm.write(f"Completed Epoch {epoch}: Val Acc = {acc:.2%}")

        # Return this to keep it visible in the live progress bar
        return {}

    optimizer = ADAM(net, cost_function, 0.0005)
    net.train(10, optimizer=optimizer, training_dataset=training_loader, epoch_callback=epoch_callback)
    #net.save("mnist_model.json")

if __name__ == "__main__":
    """net = load_from_file("mnist_example/mnist_model.json")
    net.eval_mode()
    test_data = MNIST_DataSet_PKL(type='test')
    results = [(np.argmax(net.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
    acc = sum(int(x == y) for (x, y) in results) / len(test_data)
    print(f"Test Accuracy: {acc:.2%}")"""
    
    main()