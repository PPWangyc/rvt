import torch
import torch.nn as nn
from data import FaceF_whole_video_2_labels
from torch.utils.data import DataLoader
from torchvision import transforms
from models.ST_Former import GenerateModel, RVT
from utils.utils import get_current_test_set, output_data, set_seed
from sklearn.metrics import precision_score, recall_score, f1_score
import __main__

# args parser
import argparse
argparser = argparse.ArgumentParser(description='Train and evaluate SegNet')
argparser.add_argument('--model', type=str, default='resnet34', help='Model to use')
argparser.add_argument('--batch_size', type=int, default=32, help='Batch size')
argparser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
argparser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
argparser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
argparser.add_argument('--pretrained', default=False, action='store_true', help='Use pretrained model')
argparser.add_argument('--img_path', type=str, default='/home/yanchen/Data/breathe', help='Path to the image folder')
argparser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
argparser.add_argument('--resume', type=str, default=None, help='Path to the checkpoint to resume from')
argparser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use')
argparser.add_argument('--gpu', type=int, default=0, help='GPU to use')
argparser.add_argument('--config_data', type=str, default='./configs/data.json')
argparser.add_argument('--out_dir', type=str, default='./output')
argparser.add_argument('--seed', type=int, default=0, help='Random seed')

args = argparser.parse_args()

set_seed(args.seed)

epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
momentum = args.momentum
weight_decay = args.weight_decay

best_loss = 100
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0

# define the model
net = GenerateModel()

start_epoch = 1
if args.resume:
    setattr(__main__, "RecorderMeter", GenerateModel)
    state = torch.load(args.resume)
    state_dict = state['state_dict']
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = state_dict[key]
    net.load_state_dict(new_state_dict)
    start_epoch = state['epoch'] + 1
    print('Resuming from epoch %d' % start_epoch)
    start_epoch = 1
    net = RVT(net)
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
net.to(device)

train_transform = transforms.Compose([
    transforms.Resize((112, 111)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
train_set = FaceF_whole_video_2_labels(args.img_path, train=True, transform=train_transform, config_path=args.config_data)

test_transform = transforms.Compose([
                transforms.Resize((112, 111)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
test_set = FaceF_whole_video_2_labels(args.img_path, train=False, transform=test_transform, config_path=args.config_data)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

print("train_set:", len(train_set))
print("test_set:", len(test_set))
print("batch_size:", batch_size)
print("test_set:", get_current_test_set(args.config_data))

# train the model
criterion = torch.nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

# define a function to add regularization
def add_regularization(net, weight_decay):
    l2_reg = torch.tensor(0.).to(device)
    for param in net.parameters():
        l2_reg += torch.norm(param)
    return l2_reg * weight_decay

# define the training function
def train(epochs):
    net.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) == 0:
            continue
        data, target = data.to(device), target.to(device)
        data, target = data[0], target[0]
        
        hidden = torch.zeros(1, 2).to(device)
        for i in range(len(data)):
            if i == 0 or i == len(data) - 1:
                net.train()
                optimizer.zero_grad()
                output, hidden = net(data[i], hidden)
                hidden = hidden.detach()
                if i == 0:
                    loss = criterion(output, target[0])
                else:
                    loss = criterion(output, target[1])
                print('output:', output)
                print('target:', target)
                loss = loss + add_regularization(net, weight_decay)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            else:
                with torch.no_grad():
                    output, hidden = net(data[i], hidden)

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs, batch_idx * 1, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss / (10 * batch_size)))
            running_loss = 0.0

# define the test function
def test(epoch):
    global best_loss
    global best_acc
    global total
    global best_precision
    global best_recall
    global best_f1
    global best_y_true
    global best_y_pred
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    class_acc = {}

    with torch.no_grad():
        for data, target in test_loader:
            if len(data) == 0:
                continue
            data, target = data.to(device), target.to(device)
            hidden = torch.zeros(1, 2).to(device)
            data, target = data[0], target[0]
            for i in range(len(data)):
                if i == 0 or i == len(data) - 1:
                    output, hidden = net(data[i], hidden)
                    hidden = hidden.detach()
                    y_pred.append(output.argmax(dim=1).item())
                    if i == 0:
                        y_true.append(target[0].item())
                        test_loss += criterion(output, target[0]).item()
                        correct += (output.argmax(dim=1) == target[0]).sum().item()
                        for i in range(len(target[0])):
                            if target[0][i].item() not in class_acc:
                                class_acc[target[0][i].item()] = [0, 0]
                            class_acc[target[0][i].item()][0] += 1
                            if output.argmax(dim=1)[i] == target[0][i]:
                                class_acc[target[0][i].item()][1] += 1
                        print("output:", output.argmax(dim=1))
                        print("target:", target[0])
                    else:
                        y_true.append(target[1].item())
                        test_loss += criterion(output, target[1]).item()
                        correct += (output.argmax(dim=1) == target[1]).sum().item()
                        for i in range(len(target[1])):
                            if target[1][i].item() not in class_acc:
                                class_acc[target[1][i].item()] = [0, 0]
                            class_acc[target[1][i].item()][0] += 1
                            if output.argmax(dim=1)[i] == target[1][i]:
                                class_acc[target[1][i].item()][1] += 1
                        print("output:", output.argmax(dim=1))
                        print("target:", target[1])
                    total += 1
                else:
                    output, hidden = net(data[i], hidden)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    test_loss /= total
    test_acc = 100. * correct / total
    # calculate the accuracy of each class
    for key in class_acc:
        class_acc[key] = (class_acc[key][1] / class_acc[key][0]) * 100

    print('Test set: Average loss: {:.4f} '.format(test_loss))
    print('Accuracy: {}/{} ({:.0f}%)'.format(correct, total, test_acc))
    print({"test-loss": test_loss, "test epoch": epoch, "all class accuracy": test_acc,
                "precision": precision, "recall": recall, "f1": f1,
               "class accuracy": class_acc})
    if test_acc > best_acc:
        best_loss = test_loss
        best_acc = test_acc
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_y_true = y_true
        best_y_pred = y_pred
        state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        }
        print("saving model")
        torch.save(state, './checkpoint/current.pth')

for epoch in range(start_epoch, epochs + 1):
    train(epoch)
    test(epoch)

output_dict = {"subject": get_current_test_set(args.config_data)[0], "accuracy": best_acc, "size": total,
               "precision": best_precision, "recall": best_recall, "f1": best_f1,
               "path": args.out_dir, "bal_acc": -1, "y_true": best_y_true, "y_pred": best_y_pred
               }
output_data(output_dict)