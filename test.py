import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from models import *
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=3)

    args = parser.parse_args()
    return args


def accuracy(output, labels):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(labels)
    return torch.mean(correct.float())


def main():
    args = parse_args()
    test_path = './data'
    test_file = 'fashion-mnist_test.csv'
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = FashionMNIST(test_file, test_path, transform=test_transform)

    # number of classes : 10
    num_classes = len(set(test_dataset.labels))

    # make data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # model
    model = None
    if args.version == 1:
        model = MyModel(num_classes=num_classes)  # version 1
        model.load_state_dict(torch.load('./trained_models/version1_model.pth', map_location=device))
    elif args.version == 2:
        model = EnsembleModel(num=10, num_classes=10, device=device)  # version 2
        model.load_state_dict(torch.load('./trained_models/version2_model.pth', map_location=device))
    else:
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=4)  # version 3
        model.load_state_dict(torch.load('./trained_models/version3_model.pth', map_location=device))

    model = model.to(device)

    model.eval()
    test_acc = 0.0
    print('Evaluating...')
    for samples, labels in test_loader:
        with torch.no_grad():
            samples, labels = samples.to(device), labels.to(device)
            output = model(samples)
            test_acc += accuracy(output, labels)

    print('Accuracy of the network on {} test images: {} %'.format(len(test_dataset),
                                                                   (test_acc.item() * 100.0 / len(test_loader))))


if __name__ == '__main__':
    main()
