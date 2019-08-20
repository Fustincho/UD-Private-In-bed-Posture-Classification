# Data Load
import os
import numpy as np

# PyTorch (modeling)
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Visualization
import matplotlib.pyplot as plt

# These functions are introduced along the Part 1 notebook.

# Position vectors. We load the data with respect to the file name, which is
# a number corresponding to a specific in-bed position. We take advantage of this
# and use the number to get the position with help of the following vectors.

positions_i = ["justAPlaceholder", "supine", "right",
               "left", "right", "right",
               "left", "left", "supine",
               "supine", "supine", "supine",
               "supine", "right", "left",
               "supine", "supine", "supine"]

positions_ii = {
    "B":"supine", "1":"supine", "C":"right",
    "D":"left", "E1":"right", "E2":"right",
    "E3":"left", "E4":"left", "E5":"right",
    "E6":"left", "F":"supine", "G1":"supine",
    "G2":"right", "G3":"left"
}

class_positions = ['supine', 'left', 'right', 'left_fetus', 'right_fetus']

# We also want the classes to be encoded as numbers so we can work easier when
# modeling. This function achieves so. Since left_fetus and right_fetus are not
# considered as classes in the evaluation of the original paper and since they
# are not considered in the "Experiment I", we encode them also as left and right
# positions.

def token_position(x):
  return {
      'supine': 0,
      'left': 1,
      'right': 2,
      'left_fetus': 1,
      'right_fetus': 2
  }[x]

def load_exp_i(path):
  """
  Creates a numpy array for the data and labels.

  params:
  ------
  path    -- Data path.

  returns:
  -------
  A numpy array (data, labels).
  """

  dataset = {}

  for _, dirs, _ in os.walk(path):
    for directory in dirs:
      # each directory is a subject
      subject = directory
      data = None
      labels = None
      for _, _, files in os.walk(os.path.join(path, directory)):
        for file in files:
          file_path = os.path.join(path, directory, file)
          with open(file_path, 'r') as f:
            # Start from second recording, as the first two are corrupted
            for line in f.read().splitlines()[2:]:

              raw_data = np.fromstring(line, dtype=float, sep='\t')
              # Change the range from [0-1000] to [0-255].
              file_data = np.round(raw_data*255/1000).astype(np.uint8)
              Normalize = transforms.Compose([
                          transforms.ToPILImage(),
                          transforms.ToTensor()
                          ])
              file_data = Normalize(file_data.reshape(64,32))
              file_data = file_data.view(1, 64, 32)
              # Turn the file index into position list,
              # and turn position list into reduced indices.
              file_label = token_position(positions_i[int(file[:-4])])
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)
              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

      dataset[subject] = (torch.from_numpy(data), torch.from_numpy(labels))
  return dataset

# Both air and sponge mattresses used in the data collection have a different
# size (64 x 27), opposed to the pressure mattress (64 x 32) used in the first
# experiment. Additionally, the image is rotated by 180 degrees with respect to
# the experiment one images.
# This function serves to set the format of the images equal to the ones taken
# by the pressure mat.

def resize_and_rotate(image):
  To_PIL_and_Resize = transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize((64, 32))
                      ])

  rotated = TF.rotate(To_PIL_and_Resize(image), angle=180)

  return transforms.ToTensor()(rotated)


def load_exp_ii(path):

  exp_ii_data_air = {}
  exp_ii_data_spo = {}

  # each directory is a subject
  for _, subject_dirs, _ in os.walk(path):
    for subject in subject_dirs:
      data = None
      labels = None

      # each directory is a matresss
      for _, mat_dirs, _ in os.walk(os.path.join(path, subject)):
        for mat in mat_dirs:
          for _, _, files in os.walk(os.path.join(path, subject, mat)):
            for file in files:
              file_path = os.path.join(path, subject, mat, file)
              raw_data = np.loadtxt(file_path)
              # Change the range from [0-500] to [0-255].
              file_data = np.round(raw_data*255/500).astype(np.uint8)
              file_data = resize_and_rotate(file_data)
              file_data = file_data.view(1, 64, 32)

              if file[-6] == "E" or file[-6] == "G":
                file_label = positions_ii[file[-6:-4]]
              else:
                file_label = positions_ii[file[-6]]

              file_label = token_position(file_label)
              file_label = np.array([file_label])

              if data is None:
                data = file_data
              else:
                data = np.concatenate((data, file_data), axis=0)

              if labels is None:
                labels = file_label
              else:
                labels = np.concatenate((labels, file_label), axis=0)

          if mat == "Air_Mat":
            exp_ii_data_air[subject] = (torch.from_numpy(data), torch.from_numpy(labels))
          else:
            exp_ii_data_spo[subject] = (torch.from_numpy(data), torch.from_numpy(labels))

          data = None
          labels = None

    return exp_ii_data_air, exp_ii_data_spo

### Leave-one-subject-out Cross Validation (used on Part 2)

def exp_i_cv():
  subjects_i = ["S1", "S2", "S3", "S4", "S5", "S6", "S7",
                "S8", "S9", "S10", "S11", "S12", "S13"]

  print("Performing one-subject-out cross validation on 'Experiment I':")

  torch.manual_seed(123)

  accuracies = []

  for subject in subjects_i:
    remaining_subjects = subjects_i.copy()
    remaining_subjects.remove(subject)

    trainset_exp_i = Mat_Dataset(["Base"], remaining_subjects)
    valset_exp_i = Mat_Dataset(["Base"], [subject])

    trainloader = DataLoader(trainset_exp_i, batch_size=64, shuffle=True)
    testloader = DataLoader(valset_exp_i, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model.to(device)

    epochs = 5
    running_loss = 0

    train_losses, test_losses = [], []

    for epoch in range(epochs):
      for inputs, labels in trainloader:

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

      else:

        test_loss = 0
        accuracy = 0
        model.eval()

        with torch.no_grad():
          for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            test_loss += criterion(logps, labels)

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        accuracy = accuracy/len(testloader)
        if (epoch + 1) == epochs:
          print(f"Leave out: {subject} - "
                f"Test accuracy: {accuracy:.3f}")
        running_loss = 0
        model.train()

    accuracies.append(accuracy)
  print(f"Results, one-subject-out cross validation: accuracy: {np.mean(accuracies)}")

def train_all_exp_i():
  subjects_i = ["S1", "S2", "S3", "S4", "S5", "S6", "S7",
                "S8", "S9", "S10", "S11", "S12", "S13"]

  torch.manual_seed(123)

  trainset_exp_i = Mat_Dataset(["Base"], subjects_i)

  trainloader = DataLoader(trainset_exp_i, batch_size=64, shuffle=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = CNN()

  criterion = nn.NLLLoss()

  optimizer = optim.Adam(model.parameters(), lr = 0.001)

  model.to(device)

  epochs = 5
  running_loss = 0

  train_losses = []

  for epoch in range(epochs):
    for inputs, labels in trainloader:

      inputs, labels = inputs.to(device), labels.to(device)

      optimizer.zero_grad()

      logps = model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      train_losses.append(running_loss/len(trainloader))
      if (epoch + 1) == epochs:
        print(f"Leave out: {subject} - "
              f"Test accuracy: {accuracy:.3f}")
      running_loss = 0
      model.train()

      return model, train_losses

# Custom class introduced on Part 2

class Mat_Dataset(Dataset):
  def __init__(self, mats, Subject_IDs):

    self.samples = []
    self.labels = []

    for mat in mats:
      data = datasets[mat]
      self.samples.append(np.vstack([data.get(key)[0] for key in Subject_IDs]))
      self.labels.append(np.hstack([data.get(key)[1] for key in Subject_IDs]))

    self.samples = np.vstack(self.samples)
    self.labels = np.hstack(self.labels)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, idx):
    return self.samples[idx], self.labels[idx]

# CNN model introduced in Part 2

class CNN(nn.Module):

  def __init__(self):
    super().__init__()

    ## Convolutional Layers
    #Input channels = 1, output channels = 6
    self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
    #Input channels = 6, output channels = 18
    self.conv2 = torch.nn.Conv2d(6, 18, kernel_size=3, stride=1, padding=1)

    ## Pool Layer
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    ## Mulit-Layer Perceptron
    # Hidden layers
    self.h1 = nn.Linear(18 * 16 * 8, 392)
    self.h2 = nn.Linear(392, 98)

    # Output layer, 3 neurons - one for each position
    self.output = nn.Linear(98, 3)

    # ReLU activation and softmax output
    self.relu = nn.ReLU()
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, x):

    x = x.float()
    # Add a "channel dimension"
    x = x.unsqueeze(1)

    ## Computation on convolutional and pool layers:
    # Size changes from (1, 64, 32) to (6, 64, 32)
    x = F.relu(self.conv1(x))
    # Size changes from (6, 64, 32) to (6, 32, 16)
    x = self.pool(x)
    # Size changes from (6, 32, 16) to (18, 32, 16)
    x = F.relu(self.conv2(x))
    # Size changes from (18, 32, 16) to (18, 16, 8)
    x = self.pool(x)

    # Reshape data to input to the input layer of the MLP
    # Size changes from (18, 16, 8) to (1, 2304)
    x = x.view(x.shape[0], -1)

    ## Computation on the MLP layers:
    x = self.h1(x)
    x = self.relu(x)
    x = self.h2(x)
    x = self.relu(x)
    x = self.output(x)
    x = self.logsoftmax(x)

    return x

exp_i_data = load_exp_i("dataset/experiment-i")
exp_ii_data_air, exp_ii_data_spo = load_exp_ii("dataset/experiment-ii")

datasets = {
    "Base":exp_i_data,
    "Spo":exp_ii_data_air,
    "Air":exp_ii_data_spo
}
