import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import logging

import matplotlib.pyplot as plt
import IPython.display as ipd

import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torchaudio.datasets import SPEECHCOMMANDS
import os

####################################
##   VARIABLES: CHANGE AS NEEDED  ##
####################################
train_step = 10             # number of train steps the model will do in each epoch (during Min-Min attack)
n_epoch = 4                 # number of epochs
batch_size = 256            # batch size
target_error_rate = 0.5     # loss threshold 
target_accuracy_rate = 70.0 # accuracy threshold
eps = 0.009                 # epsilon
ex_name = "experiments"     # folder name to save model to
# Used in Testing/debugging
SR = 16000
EXAMPLES = 5




###############################################################################
## SET UP/LOAD DATASETS
###############################################################################

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


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

#plt.plot(waveform.t().numpy());
# Contains names of all sound labels
label_types = sorted(list(set(datapoint[2] for datapoint in train_set)))
new_sample_rate = 8000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

#ipd.Audio(transformed.numpy(), rate=new_sample_rate)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(label_types.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    return label_types[index]


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


#batch_size = 256

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









#####################################################################################################
## DEFINE the Network (CNN)
#####################################################################################################
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=17):
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

# Set model, send to GPU, and print
model = M5(n_input=transformed.shape[0], n_output=len(label_types))
model.to(device)
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

#optimizer (Adam) and scheduler (stepLR)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

#### Criterion (cross entropy loss) and init random noise####
criterion = nn.CrossEntropyLoss()
noise_shape = [len(train_set), 16000]
random_noise = torch.zeros(noise_shape) #init with all zeroes















#######################################################################################################
### ADD PERTURBATION/NOISE ###
#######################################################################################################
def perturb_eval(random_noise, train_loader, model, startAndEnd_list, mask_cord_list):
    print("In Perturb Eval", flush=True)
    loss_meter = AverageMeter()
    err_meter = AverageMeter()
   # model.eval()
    model = model.to(device)
    idx_v = 0
    # Iterate over Data Loader (batches of audio and labels)
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device,non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i, (datum, label) in enumerate(zip(data, labels)):
                if not torch.is_tensor(random_noise):
                    sample_noise = torch.tensor(random_noise[idx_v]).to(device)
                else:
                    sample_noise = random_noise[idx_v].to(device)
                length = datum.shape[1] # CHANGED, was 0
                mask = np.zeros(length, np.float32)
                start,end = startAndEnd_list[idx_v]
                mask[start:end] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                data[i] = data[i] + sample_noise
                idx_v += 1
        # squeeze to get rid of 2nd dimension
        pred = model(data).squeeze(1)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err/len(labels))
    return loss_meter.avg, err_meter.avg
                

def min_min_attack(data, labels, model, optimizer, criterion, random_noise):
    perturbed_data = data.clone()
    eta = random_noise.clone().detach().requires_grad_(True).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    optimizer.zero_grad()
    
    outputs = model(perturbed_data + eta)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    
    # Update the noise
    eta_grad = eta.grad.sign()
    eta = eta - eps * eta_grad
    eta = torch.clamp(eta, -eps, eps)
    
    perturbed_data = data + eta
    return perturbed_data, eta


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


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
 

idx_temp = 0
mask_cord_list = []
startAndEnd_list = []
idx = 0

## Go through all batches of sound data and labels
for data, labels in train_loader:
    ## Go through each sound 
    for i, (datum,label) in enumerate(zip(data,labels)):
        # For each image, generate noise
        noise = random_noise[idx].numpy() #(Maybe no on numpy CHANGED)
        mask_cord, startAndEnd = patch_noise_to_sound(noise, waveform_length=datum.shape[1], segment_location='center') 
    #    print(f"Single Mask cord: {mask_cord}")
        # Convert the mask back to torch tensor and apply it to the datum
 ## !      noisy_waveform = datum.numpy() + mask_cord
 ## !      noisy_waveform = torch.tensor(noisy_waveform)
        
        ## CHANGED, Ue noisy_waveform for training
  ##!      data[i] = noisy_waveform
        startAndEnd_list.append(startAndEnd)    #hold starts and end indicies
        mask_cord_list.append(mask_cord)        #holds masks for each audio sample (?)
        idx += 1
 











##########################################################
### TRAIN MODEL ON PERTURBATION ###
##########################################################

## Training phase for MIN-MIN Attack: applies noise to each sound, then trains
## the model on the noisy sounds
print("Start Training/Min-min", flush=True)
condition = True
train_idx = 0
data_iter = iter(train_loader) #to loop over dataset in batches
#logger.info('=' * 20 + 'Searching Samplewise Perturbuations' + '=' * 20)
print('=' * 20 + 'Searching Samplewise Perturbuations' + '=' * 20, flush=True)

while condition:
    #for j in tqdm(range(0,train_step), total=train_step):
    for j in tqdm(range(train_step)):
        ## Attempt to load next batch of sounds/labels
        try:
            (data,labels) = next(data_iter)
        except: 
            train_idx = 0
            data_iter = iter(train_loader)
            (data,labels) = next(data_iter)

        # Move data to device and add noise
        data,labels = data.to(device), labels.to(device)

        #Train the model on noisy data
        for i, (datum,label) in enumerate(zip(data,labels)): #iterate over sound in current batch
            sample_noise = random_noise[train_idx].numpy()
            mask_cord = mask_cord_list[train_idx]
            datum_cpu = datum.cpu().numpy()
            noisy_waveform = datum_cpu + mask_cord #_cpu
            noisy_waveform = torch.tensor(noisy_waveform).to(device)

            data[i] = noisy_waveform 
            train_idx += 1

        # Train the model on NOISY DATA
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(),labels)
        loss.backward()
        optimizer.step()

    ## Seach for perturbations (noise) and update noise on min-min
    idx = 0
    for i, (data,labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)
        ## MAYBE ADD ^ model = images.to(model)
        ## maybe add batch_start_idx = idx
    batch_noise, batch_start_idx = [], idx
    # Iterate over audio in current batch
    for j, datum in enumerate(data):
        sample_noise = random_noise[idx].numpy()
        mask_cord = mask_cord_list[idx]
        datum_cpu = datum.cpu().numpy() #CHANGED 2

      #  noisy_waveform = datum.numpy() + mask_cord
        noisy_waveform = datum_cpu + mask_cord #CHANGED 2
        noisy_waveform = torch.tensor(noisy_waveform).to(device)
        batch_noise.append(noisy_waveform)
        idx += 1
    # MIGHT need this line to have .to(device), might need to be after requires_grad below
    batch_noise = torch.stack(batch_noise)

    #Eval the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    #loss_avg, error_rate = perturb_eval(random_noise, test_loader_loader, )
    ## MIN-MIN Attack
    perturb_audio, eta = min_min_attack(data, labels, model, optimizer, criterion, random_noise=batch_noise)
    for i, delta in enumerate(eta):
        mask_cord = mask_cord_list[batch_start_idx + i]
        #delta = delta.numpy()
        delta_cpu = delta.detach().cpu().numpy() # move delta tensor to CPU
        random_noise[batch_start_idx + i] = torch.tensor(delta_cpu).to(device) #Was just delta
    #print("Before EVAL...", flush=True)
    loss_avg, error_rate = perturb_eval(random_noise, train_loader,model,startAndEnd_list=startAndEnd_list, mask_cord_list=mask_cord_list)
    #logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate * 100))
    print('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate * 100), flush=True)

   # Check if threshold is under error_rate
   # if error_rate < target_error_rate:
   #     condition = False

    # Check if threshold is over accuracy
    if (100-error_rate * 100) > target_accuracy_rate:
        condition = False
        print("Current working directory:", os.getcwd(), flush=True)
        for i, waveform in enumerate(perturb_audio[:EXAMPLES]):
            waveform = waveform.cpu()
            # Check the shape of the waveform
            torchaudio.save(f'perturbations_sample_{i}.wav', waveform, SR)
## END OF CONDITION LOOP











###############################################
### UPDATE NOISE, SAVE MODEL ###
###############################################

# Finale Noise Update to Audio
if torch.is_tensor(random_noise):
    new_random_noise = []
    # Iterate over random_noise
    for idx in range(len(random_noise)):
        sample_noise = random_noise[idx]
        # Get waveform length and create zero mask of same size
        waveform_length = data.shape[2] #CHANGED, was [1]
        mask = np.zeros((waveform_length), np.float32)
        # Get coords of segment and ple noise in location
        start,end = startAndEnd_list[idx]
        mask[start:end] = sample_noise.cpu().numpy()
        #Conver back to tensor and add to list
        new_random_noise.append(torch.from_numpy(mask))
    # Stack list of noises tensors into single tensor
    new_random_noise=torch.stack(new_random_noise)
    random_noise = new_random_noise
#else: random noise isnt a tensor, dont change it
    
## Save the Noisy Model
torch.save(random_noise, os.path.join(ex_name, 'perturbation.pt'))
print(noise)
print(noise.shape)
print('Noise saved at %s ' % (os.path.join(ex_name, 'perturbation.pt')), flush=True)






#################################################################
## REFERENCES ##
#################################################################
# 1) https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
# 2) https://github.com/HanxunH/Unlearnable-Examples
#################################################################
