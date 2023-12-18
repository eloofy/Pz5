import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from NN import SimpleNet

class Trainer:
    def __init__(self, num_epochs=5, batch_size=64, learning_rate=0.01, momentum=0.9):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.valloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        self.net = SimpleNet(3, 32, 10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.train_accuracy_list, self.train_recall_list, self.train_precision_list, self.train_f1_list = [], [], [], []
        self.val_accuracy_list, self.val_recall_list, self.val_precision_list, self.val_f1_list = [], [], [], []

    def calculate_metrics(self, loader, model):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro')
        return accuracy, recall, precision, f1

    def train_and_evaluate(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            self.net.train()
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            train_metrics = self.calculate_metrics(self.trainloader, self.net)
            val_metrics = self.calculate_metrics(self.valloader, self.net)

            train_acc, train_rec, train_prec, train_f1 = train_metrics
            val_acc, val_rec, val_prec, val_f1 = val_metrics

            self.train_accuracy_list.append(train_acc)
            self.train_recall_list.append(train_rec)
            self.train_precision_list.append(train_prec)
            self.train_f1_list.append(train_f1)

            self.val_accuracy_list.append(val_acc)
            self.val_recall_list.append(val_rec)
            self.val_precision_list.append(val_prec)
            self.val_f1_list.append(val_f1)

        print('Finished Training')

    def plot_metrics(self, metric_name, title):
        epochs = range(1, self.num_epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, getattr(self, f'train_{metric_name}_list'), label=f'Train {metric_name}')
        plt.plot(epochs, getattr(self, f'val_{metric_name}_list'), label=f'Validation {metric_name}')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_and_evaluate()
    trainer.plot_metrics('accuracy', 'Training and Validation Accuracy')
    trainer.plot_metrics('recall', 'Training and Validation Recall')
    trainer.plot_metrics('precision', 'Training and Validation Precision')
    trainer.plot_metrics('f1', 'Training and Validation F1 Score')
