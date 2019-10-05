import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from models import *
import copy


def accuracy(output, labels):
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(labels)
    return torch.mean(correct.float())


def main():
    # aply transforms to return img as tensor type
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_path = './data'
    train_file = 'fashion-mnist_train.csv'
    test_path = './data'
    test_file = 'fashion-mnist_test.csv'
    train_dataset = FashionMNIST(train_file, train_path, transform=train_transform)
    test_dataset = FashionMNIST(test_file, test_path, transform=test_transform)

    # number of classes : 10
    num_classes = len(set(train_dataset.labels))

    # make dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # model = MyModel(num_classes=num_classes) # version 1
    model = EnsembleModel(num=10, num_classes=10, device=device)  # version 2
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000, 15000], gamma=0.5)

    total_loss, total_acc = 0, 0
    loss_list = []
    acc_list = []

    epochs = 100
    itr = 1
    p_itr = 1000

    # start training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for samples, labels in train_loader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += accuracy(output, labels)
            scheduler.step()

            if itr % p_itr == 0:
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, train_acc: {:.3f}'.format(epoch + 1, epochs,
                                                                                                   itr, total_loss / p_itr,
                                                                                                   total_acc / p_itr))
                loss_list.append(total_loss / p_itr)
                acc_list.append(total_acc / p_itr)
                total_loss, total_acc = 0, 0
            itr += 1

        model.eval()
        test_acc = 0.0
        for samples, labels in test_loader:
            with torch.no_grad():
                samples, labels = samples.to(device), labels.to(device)
                output = model(samples)
                test_acc += accuracy(output, labels)

        print('Accuracy of the network on {} test images: {}%'.format(len(test_dataset),
                                                                      round(test_acc.item()*100.0/len(test_loader), 2)))

        if (test_acc.item() > best_acc):
            best_acc = test_acc.item()
            best_model_wts = copy.deepcopy(model.state_dict())
            print('update best')

        print('-' * 10)

    print('best acc on test set: ', best_acc/len(test_loader))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'model3.pth')


if __name__ == '__main__':
    main()
