import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

input_size = 1
output_size = 1
num_epochs = 600
learning_rate = 0.001

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694]], dtype=np.float32)


class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LR(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(epoch + 1, loss.data.numpy())

predicted = model(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

print(model.state_dict())

torch.save(model.state_dict(), 'model.pt')
