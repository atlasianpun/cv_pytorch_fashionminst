### Project Title
Image Recognition using PyTorch

### Project Description
This project implements an image recognition model using PyTorch, focusing on the FashionMNIST dataset. The model is designed to classify images of fashion items into one of ten categories. It demonstrates essential deep learning techniques such as data loading, model definition, training, and evaluation using a neural network.

### Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Credits](#credits)
7. [License](#license)

### Installation
To install the required dependencies, run the following commands:
```bash
pip3 uninstall --yes torch torchaudio torchvision torchtext torchdata
pip3 install torch torchaudio torchvision torchtext torchdata
```

### Usage
1. **Download and prepare the dataset:**
   ```python
   from torchvision import datasets
   from torchvision.transforms import ToTensor

   training_data = datasets.FashionMNIST(
       root="data",
       train=True,
       download=True,
       transform=ToTensor(),
   )

   test_data = datasets.FashionMNIST(
       root="data",
       train=False,
       download=True,
       transform=ToTensor(),
   )
   ```

2. **Create data loaders:**
   ```python
   from torch.utils.data import DataLoader

   batch_size = 64
   train_dataloader = DataLoader(training_data, batch_size=batch_size)
   test_dataloader = DataLoader(test_data, batch_size=batch_size)
   ```

3. **Define the model:**
   ```python
   import torch
   from torch import nn

   class NeuralNetwork(nn.Module):
       def __init__(self):
           super().__init__()
           self.flatten = nn.Flatten()
           self.linear_relu_stack = nn.Sequential(
               nn.Linear(28*28, 512),
               nn.ReLU(),
               nn.Linear(512, 512),
               nn.ReLU(),
               nn.Linear(512, 256),
               nn.ReLU(),
               nn.Linear(256, 10),
           )

       def forward(self, x):
           x = self.flatten(x)
           logits = self.linear_relu_stack(x)
           return logits

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = NeuralNetwork().to(device)
   ```

4. **Train the model:**
   ```python
   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

   def train(dataloader, model, loss_fn, optimizer):
       size = len(dataloader.dataset)
       model.train()
       for batch, (X, y) in enumerate(dataloader):
           X, y = X.to(device), y.to(device)
           pred = model(X)
           loss = loss_fn(pred, y)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
           if batch % 100 == 0:
               loss, current = loss.item(), (batch + 1) * len(X)
               print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

   def test(dataloader, model, loss_fn):
       size = len(dataloader.dataset)
       num_batches = len(dataloader)
       model.eval()
       test_loss, correct = 0, 0
       with torch.no_grad():
           for X, y in dataloader:
               X, y = X.to(device), y.to(device)
               pred = model(X)
               test_loss += loss_fn(pred, y).item()
               correct += (pred.argmax(1) == y).type(torch.float).sum().item()
       test_loss /= num_batches
       correct /= size
       print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

   epochs = 100
   for t in range(epochs):
       print(f"Epoch {t+1}\n-------------------------------")
       train(train_dataloader, model, loss_fn, optimizer)
       test(test_dataloader, model, loss_fn)
   print("Done!")
   ```

### Model Architecture
The neural network model consists of:
- A flatten layer to convert 2D images into 1D tensors.
- Three fully connected (linear) layers with ReLU activation functions.
- An output layer with 10 units (one for each class).

### Training and Evaluation
- **Training:** The model is trained using stochastic gradient descent (SGD) with cross-entropy loss.
- **Evaluation:** The model's performance is evaluated on the test dataset, reporting accuracy and average loss.

### Results
After training for 100 epochs, the model achieves a certain level of accuracy on the test dataset, which is printed at the end of each epoch.

### Credits
- The project is developed using PyTorch, an open-source deep learning framework.

### License
This project is licensed under the MIT License.
