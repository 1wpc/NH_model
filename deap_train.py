import torch
from deep_dataset import DeapDataset
from torch.utils.data import Dataset, DataLoader

from net import Net


train_dataset = DeapDataset(flag='train')
test_dataset = DeapDataset(flag='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Net(num_classes=1, in_channels=8064, grid_size=(4, 6))
lossfun = torch.nn.BCELoss()
opter = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def evaluate():
    correct=0
    total=0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x,labels=data
            x=x.to(device)
            labels=labels.to(device)
            outputs=model(x)
            predict=(outputs >= 0.5).int().flatten()
            total+=labels.size(0)
            correct+=(predict==labels).sum().item()
    return 100*correct/total

for epoch in range(epochs):
    running_loss=0.0
    for i, data in enumerate(train_loader,0):
        inputs, labels=data
        inputs=inputs.to(device)
        labels=labels.to(device)
        opter.zero_grad()
        outputs=model(inputs)#输入数据进网络
        loss=lossfun(outputs.flatten(),labels.float())
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

torch.save(model.state_dict(), './model/deap_model_valence.pth')