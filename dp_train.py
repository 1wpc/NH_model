import torch
from dp_dataset import ModmaDataset
from torch.utils.data import Dataset, DataLoader

from net import Net

train_dataset = ModmaDataset(path='./modma', flag='train')
test_dataset = ModmaDataset(path='./modma', flag='test')

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

model = Net(num_classes=1, in_channels=15000, grid_size=(4, 6))
# lossfun = torch.nn.CrossEntropyLoss()
lossfun = torch.nn.BCELoss()
opter = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def evaluate():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            data,labels=data
            data=data.to(device)
            labels=labels.to(device)
            outputs=model(data)
            predict=(outputs >= 0.5).int().flatten()
            total+=labels.size(0)
            correct+=(predict==labels).sum().item()
    return 100*correct/total

for epoch in range(epochs):
    running_loss=0.0
    for i, data in enumerate(train_loader,0):
        inputs, lables=data
        inputs=inputs.to(device)
        lables=lables.to(device)
        opter.zero_grad()
        outputs=model(inputs)#输入数据进网络
        loss=lossfun(outputs.flatten(),lables.float())
        loss.backward()
        opter.step()
        running_loss+=loss.item()
        print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss))
        running_loss=0.0
        # if i % 10==0 and i!=0:
        #     print('[%d, %5d] loss: %.3f'%(epoch+1,i+1,running_loss/10))
        #     running_loss=0.0
    print(evaluate())
print('finish')

print(evaluate())

torch.save(model.state_dict(), './model/dp_model.pth')