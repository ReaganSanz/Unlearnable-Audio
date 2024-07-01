import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import mlconfig
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
from tqdm import tqdm

###################################
## VARIABLES ##
###################################
num_classes = 13 # CHANGE DEPENDING ON NUM CLASSES
batch_size = 256
perturb_tensor_path = "experiments/perturbation.pt"
log_interval = 20
n_epoch = 4
poison_rate = 1.0


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]



####################################
# DEFINE M5 MODEL #
####################################
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)




###############################################################################
## SET UP/LOAD DATASETS
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

   
# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

#plt.plot(waveform.t().numpy());
    
# Contains names of all sound labels
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

#ipd.Audio(transformed.numpy(), rate=new_sample_rate)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


#   batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

## Set model to our custom M5 model
model = M5(n_input=transformed.shape[0], n_output=len(labels)).to(device)

## Set datasets with poisoned data
#    perturb_tensor = torch.load(perturb_tensor_path, map_location=device)

########################################
#### ADD NOISY SAMPLES ###
########################################
def patch_noise_to_sound(noise, waveform_length=16000, segment_location='center'):
    # Init mask to zeroes
    mask = np.zeros(waveform_length, dtype=np.float32) #might be dtype=np.float32
    noise_length = noise.shape[0]
    if segment_location == 'center' or (waveform_length == noise_length):
        # Apply noise to the center of the waveform
        start = (waveform_length - noise_length) // 2
    elif segment_location == 'random':
        # Apply noise to a random location in the waveform
        start = np.random.randint(0, waveform_length - noise_length)
    else:
        raise ValueError('Invalid segment location')

    # Find end position (start+length), then check if its in bounds
    end = start + noise_length
    mask[start:end] = noise
    return mask, (start, end)
    
class PoisonSC(SubsetSC):
    def __init__(self, subset, poison_rate=1.0, perturb_tensor_filepath=None, patch_location='center'):
        super().__init__(subset=subset)
        # Load Noise from pertubation.pt, set variables
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
      #  print(f"Perturb Tensor shape: {self.perturb_tensor.shape}", flush=True)
        self.perturb_tensor = self.perturb_tensor.cpu().numpy()
        self.patch_location = patch_location
        self.poison_rate = poison_rate  # Percent of data that is poisoned
        self.poisoned_samples = {} #to store modified examples
        # Apply Noise to samples so [poison_rate]% of them are noisy
        # Randomly selected poison targets
        targets = list(range(len(self)))
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        temp_idx = 0
        for idx in self.poison_samples_idx:
            noise = self.perturb_tensor[idx % len(self.perturb_tensor)]
         #   print(f"{idx+1}\n", flush=True)
         #   print(f"Noise shape: {noise.shape}", flush=True)
            noise, (start,end) = patch_noise_to_sound(noise, waveform_length=16000, segment_location=self.patch_location)
         #   print(f"Noise after patch_noise_to_sound shape: {noise.shape}", flush=True)
            waveform, sample_rate, label, *_ = self[idx]
            waveform = self._standardize_waveform(waveform)  # if waveforms are incorrect sizes
          #  print(f"Waveform shape before adding noise: {waveform.shape}", flush=True)

            num_channels = waveform.shape[0]
            poisoned_waveform = np.zeros_like(waveform.numpy())
            for channel in range(num_channels):
                poisoned_waveform[channel] = np.clip(waveform[channel].numpy() + noise, -1, 1)
           # print(f"Poisoned waveform shape: {poisoned_waveform.shape}", flush=True)
            ##TESTING
            if temp_idx < 3:
                original_path = os.path.join('./testing/', f'original_sample_{idx}.wav')
                poisoned_path = poisoned_path = os.path.join('./testing/', f'poisoned_sample_{idx}.wav')
                torchaudio.save(original_path, waveform, sample_rate)
                torchaudio.save(poisoned_path, torch.tensor(poisoned_waveform), sample_rate)

            temp_idx += 1
            self.poisoned_samples[idx] = (torch.tensor(poisoned_waveform), sample_rate, label)

    def __getitem__(self,idx):
        if idx in self.poisoned_samples:
            return self.poisoned_samples[idx]
        else:
           # print("Else", flush=True)
            return super().__getitem__(idx)
         
    # To set clean data to have a constant waveform.shape
    def _standardize_waveform(self, waveform, target_length=16000):
        waveform_length = waveform.shape[1]
        if waveform_length < target_length:
            # Pad waveform with zeros if it's shorter than target_length
            padding = target_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform_length > target_length:
            # Truncate waveform if it's longer than target_length
            waveform = waveform[:, :target_length]
        return waveform
    
poison_train_set = PoisonSC("training", poison_rate=poison_rate, perturb_tensor_filepath=perturb_tensor_path)
poison_train_loader = DataLoader(poison_train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Print dataset information
print(f'Poisoned Training Dataset: {len(poison_train_set)} samples')
print(f'Test Dataset: {len(test_set)} samples')





##########################
## TRAIN AND TEST ##
##########################
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(poison_train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"============Train Epoch: {epoch} [{batch_idx * len(data)}/{len(poison_train_loader.dataset)} ({100. * batch_idx / len(poison_train_loader):.0f}%)]\tLoss: {loss.item():.6f}================")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n", flush=True)
    return 100. * correct / len(test_loader.dataset)

pbar_update = 1 / (len(poison_train_loader) + len(test_loader))
losses = []
final_acc = 0.0
# transform on same device as model/data
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        final_acc = test(model, epoch)
        scheduler.step()

# Plot the training loss and Print the final accuracy/varaibles
plt.plot(losses)
plt.title("training loss")
plt.savefig(f'experiments/plot.png')
plt.close()


print("=== VARIABLES ===", flush=True)
print(f"Epoch: {n_epoch}\n Batch_size: {batch_size}\n num_classes: {num_classes}\n Log_Interval: {log_interval}", flush=True)
print(f"Final Accuracy: {final_acc:.2f}%", flush=True)

