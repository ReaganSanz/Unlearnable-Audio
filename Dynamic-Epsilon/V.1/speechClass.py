import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import logging
import random

import matplotlib.pyplot as plt
import IPython.display as ipd

import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from torchaudio.datasets import SPEECHCOMMANDS
import os

####################################
##   VARIABLES: CHANGE AS NEEDED  ##
####################################
# Generating Noise Variables
batch_size = 256            # batch size (increase to speed up job) as 256
target_error_rate = 0.1    # loss threshold (CURRENTLY USING)
target_accuracy_rate = 90.0 # accuracy threshold increase
# Dynamic Epsilon / Min-min Varaibles
eps_values = [0.005, 0.04, 0.09, 0.1, 0.2]           # epsilon   increase, make sure noise within range
eps_cutoff = [0, 0.025, 0.05, 0.1, 0.3]             # determines segments' corresponding eps values
step_size_factor = 25          # distance of each step (in min-min attack)
segment_size = 1600
train_step = 20             # number of train steps the model will do in each epoch (during Min-Min attack) increase to raise unlearnability
# Audio Sample Varaibles
seed = 8
transform_sample_rate = 8000
ex_name = "experiments"     # folder name to save model to
# Testing/debugging Variables
SR = 16000
EXAMPLES = 3




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
            random.seed(seed)
            random.shuffle(self._walker)


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")


#Shuffle indices
train_sampler = SubsetRandomSampler(torch.randperm(len(train_set)))


# Testing first dataset sample
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print ("==Test [0]==", flush=True)
print(f"Testset[0]: {train_set[0]}")
print(f" Waveform {waveform} \n Sample Rate: {sample_rate} \n Label: {label} \n Utterance Num: {utterance_number}", flush=True)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
# Save the waveform as a .wav file
##torchaudio.save("test-testing/FIRSTSAMPLE1.wav", waveform, sample_rate)


# Contains names of all sound labels
label_types = sorted(list(set(datapoint[2] for datapoint in train_set)))

# TRANSFORMATIONS
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=transform_sample_rate)
transformed = transform(waveform)


# Normalize to [-1, 1] range
transformMin = transformed.min().item()
transformMax = transformed.max().item()
transformed = 2 * (transformed - transformMin) / (transformMax - transformMin) - 1
print("Minimum value of waveform:", transformed.min().item(), flush=True)
print("Maximum value of waveform:", transformed.max().item(), flush=True)



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
    shuffle=False,      #CHANGED TO FALSE!!!
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
#print(f"Transformed shape 2 = {transformed.shape[0]}")
# Transformed Shape = 1
model = M5(n_input=transformed.shape[0], n_output=len(label_types))
model.to(device)
print(model)

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
            
                length = datum.shape[1] 
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
                

class PerturbationTool:
    def __init__(self, epsilon_values, epsilon_cutoff, segment_size, step_size_factor, num_steps,seed=0):
        self.epsilon_vals = epsilon_values
        self.epsilon_cutoff = epsilon_cutoff
        self.seg_size = segment_size
        self.step_size_fac = step_size_factor
        self.num_steps = num_steps
        self.seed = seed
        np.random.seed(seed)
    
    def dynamic_eps(self, amp):

        if amp < self.epsilon_cutoff[1]:
            epVal = self.epsilon_vals[0]
        elif amp < self.epsilon_cutoff[2]:
            epVal = self.epsilon_vals[1]
        elif amp < self.epsilon_cutoff[3]:
            epVal = self.epsilon_vals[2]
        elif amp < self.epsilon_cutoff[4]:
            print("3",flush=True)
            epVal = self.epsilon_vals[3]
        else:
            epVal = self.epsilon_vals[4]
        return epVal

    def min_min_attack(self, audio_samples, labels, model, optimizer, criterion, i, random_noise=None):

        device = audio_samples.device
        current_batch_size, num_channels, audio_len = audio_samples.shape

        # Segment the audio samples
        num_segments = audio_len // self.seg_size
        
        if audio_len % self.seg_size != 0:
            num_segments += 1  # Handle last partial segment

        perturb_audio = audio_samples.clone().detach().to(device)
        perturb_audio.requires_grad = True

        eta = torch.zeros_like(audio_samples).to(device)
        # Init the list of perturb_audio and eps for each segment in each audio sample
        segment_noise_list = [[] for _ in range(current_batch_size)]
        segment_eps_list = [[] for _ in range(current_batch_size)]

        ## 1: Go through each segment, finding epsilon values ##
        for ind in range(num_segments):
            startSeg = ind*self.seg_size
            endSeg = min((ind+1) * self.seg_size, audio_len)
            
            # Go through each sample in batch, finding epsilon values 
            for b in range(current_batch_size):
                segment = perturb_audio[b:b+1,:, startSeg:endSeg]
            
                #### OPTION 1: Find Mean to detmermine epsilon for each segment ####
                ## Get mean amplitude value of the current segment to determine eps 
                # mean_amp = segment.abs().mean().item()
                # epsilon = self.dynamic_eps(mean_amp) # returns eps for this segment

                #### OPTION 2: Find the n-th percentile to determine epsilon for each segment ####
                n_percent = 0.50
                perc_amp = torch.quantile(segment.abs(), n_percent).item()
                epsilon = self.dynamic_eps(perc_amp)
        

                #### OPTION 3: Use *energy* to determine epsilon for each segment ####
                # UNFINISHED. TO DO

                # Set step_size based on epsilon and global factor 
                step_size = epsilon/self.step_size_fac

                # Init Noise
                if random_noise is None:
                    segment_noise = torch.FloatTensor(segment.shape).uniform_(-epsilon, epsilon).to(device)
                else:
                    segment_noise = random_noise[b:b+1, :, startSeg:endSeg]
                
                # Put eps and perturb values into their lists
                segment_perturb = Variable(segment.data + segment_noise, requires_grad=True)
                segment_perturb = Variable(torch.clamp(segment_perturb, -1, 1), requires_grad=True) 
                segment_noise_list[b].append((segment_perturb, segment_noise, epsilon, step_size, startSeg, endSeg))
                segment_eps_list[b].append(epsilon) 
        
        # Testing
        if i == 0: 
            print(f"EPSILON 1: {segment_eps_list[0]}", flush=True)
            print(f"EPSILON 2: {segment_eps_list[1]}", flush=True)
            print(f"EPSILON 3: {segment_eps_list[2]}", flush=True)
        # print(f"Segment Noise List shape: {len(segment_noise_list)}", flush=True)
            # FOR TESTING:
            '''
            for sample_idx in range(3):
                plt.figure(figsize=(10, 4))
                plt.plot(audio_samples[sample_idx, 0].cpu().numpy())
                plt.title(f'Sample {sample_idx + 1} Waveform')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.savefig(f'epPlot/sample_{sample_idx + 1}_waveform.png')
                plt.close()
            '''



        ## 2:  Go through num_steps times, Updating noise across ENTIRE wavelength ##
        for _ in range(self.num_steps):
            full_perturb_audio = []
        
            # Update Each Segment Perturbation
            for b in range(current_batch_size): # For number of elements in current batch
                sample_perturb_audio = []
                for j, (segment_perturb, segment_noise, epsilon, step_size, startSeg, endSeg) in enumerate(segment_noise_list[b]):
                    sample_perturb_audio.append(segment_perturb)
                sample_perturb_audio = torch.cat(sample_perturb_audio, dim=2)
                full_perturb_audio.append(sample_perturb_audio)

            # Concatenate the segments to form the full perturbed audio.
            full_perturb_audio = torch.cat(full_perturb_audio, dim=0) # was dim = 2
            full_perturb_audio = torch.tensor(full_perturb_audio.detach(), requires_grad=True).to(device)
            opt = torch.optim.SGD([full_perturb_audio], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()

            # Calculate Logits and loss for the *entire* perturbed audio (NOT by segment)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(full_perturb_audio)
                logits = logits.squeeze(1)  # to get rid of extra dimension. 
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, full_perturb_audio, labels, optimizer)
            
            loss.backward(retain_graph=True) 

            # Update Each segment based on loss of the combined segments
            for b in range(current_batch_size):
                for j, (segment_perturb, segment_noise, epsilon, step_size, startSeg, endSeg) in enumerate(segment_noise_list[b]):
                    grad_segment = full_perturb_audio.grad[b:b+1, :, startSeg:endSeg]
                    if grad_segment is not None:
                        eta_segment = step_size * grad_segment.data.sign() * (-1)
                        segment_perturb = Variable(segment_perturb.data + eta_segment, requires_grad=True)
                        eta_segment = torch.clamp(segment_perturb.data - audio_samples[b:b+1, :, startSeg:endSeg].data, -epsilon, epsilon)
                        segment_perturb = Variable(audio_samples[b:b+1, :, startSeg:endSeg].data + eta_segment, requires_grad=True)
                        segment_perturb = Variable(torch.clamp(segment_perturb, -1, 1), requires_grad=True)

                        # Update the noise list and eta
                        segment_noise_list[b][j] = (segment_perturb, segment_noise, epsilon, step_size, startSeg, endSeg)
                        eta[b:b+1, :, startSeg:endSeg] = eta_segment


        # Update the overall perturbed audio and return
        new_perturb_audio = perturb_audio.clone().detach()
        for b in range(current_batch_size):
            for segment_perturb, _, _, _, start, end in segment_noise_list[b]:
                new_perturb_audio[b:b+1, :, start:end] = segment_perturb.detach() #NEW
        
        return new_perturb_audio, eta


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
    mask = np.zeros(waveform_length, dtype=np.float32)  
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

print("Right before train_loader", flush=True)
## Go through all batches of sound data and labels
for data, labels in train_loader:
    ## Go through each sound 
    for i, (datum,label) in enumerate(zip(data,labels)):
        # For each image, generate noise
        noise = random_noise[idx].numpy() 
        mask_cord, startAndEnd = patch_noise_to_sound(noise, waveform_length=datum.shape[1], segment_location='center') 
        startAndEnd_list.append(startAndEnd)    #hold starts and end indicies
        mask_cord_list.append(mask_cord)        #holds masks for each audio sample 
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
clean_waveform_list =[]
noisy_waveform_list = []
#logger.info('=' * 20 + 'Searching Samplewise Perturbuations' + '=' * 20)
print('=' * 20 + 'Searching Samplewise Perturbuations' + '=' * 20, flush=True)

# Do while threshold has not been met
while condition:
    ## Step 1: Iterate though Batches and it's data- adding noise and training
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

        ## 1: Add noise to each sample
        for i, (datum,label) in enumerate(zip(data,labels)): #iterate over sound in current batch
            sample_noise = random_noise[train_idx]
            mask = np.zeros(datum.shape[1], np.float32)
            start,end = startAndEnd_list[train_idx]
            mask[start:end] = sample_noise.cpu().numpy()
            sample_noise = torch.from_numpy(mask).to(device)
            data[i] = data[i] + sample_noise
            train_idx += 1
           

        ## 2: Train the batch on NOISY DATA
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        # Train batch below...
        model.zero_grad()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(),labels)
        loss.backward()
        optimizer.step()

    ## STEP 2: Seach for perturbations (noise) and update noise on min-min
    idx = 0
    for i, (data,labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)
        batch_noise, batch_start_idx = [], idx
        # Iterate over audio in current batch
        for j, datum in enumerate(data):
            sample_noise = random_noise[idx]
            mask = np.zeros(datum.shape[1], np.float32)
            start,end = startAndEnd_list[idx]
            mask[start:end] = sample_noise.cpu().numpy()
            sample_noise = torch.from_numpy(mask).to(device)
        
            datum_cpu = datum.cpu().numpy()
            clean_waveform_list.append(datum_cpu)  # Add clean waveform to list

            noisy_waveform = datum_cpu + mask  # Add noise to waveform
            noisy_waveform = torch.tensor(noisy_waveform).to(device)
            batch_noise.append(noisy_waveform)
            idx += 1

        #Eval the model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        ## MIN-MIN Attack
        batch_noise = torch.stack(batch_noise)
        attack = PerturbationTool(eps_values, eps_cutoff, segment_size, step_size_factor, train_step)
        perturb_audio, eta = attack.min_min_attack(data, labels, model, optimizer, criterion, i, random_noise=batch_noise)

        ## OUTPUT is perturb_audio and eta (eta = delta, perturb is x+eta)
        for i, delta in enumerate(eta):
            mask_cord = mask_cord_list[batch_start_idx + i]
            delta_cpu = delta.detach().cpu().numpy() # move delta tensor to CPU
            random_noise[batch_start_idx + i] = torch.tensor(delta_cpu).to(device) #Was just delta


    loss_avg, error_rate = perturb_eval(random_noise, train_loader,model,startAndEnd_list=startAndEnd_list, mask_cord_list=mask_cord_list)
    print('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate * 100), flush=True)

   # Check if threshold is under error_rate
   # if error_rate < target_error_rate:
   #     condition = False

    # Check if threshold is over accuracy OR under loss
    #if (100-error_rate * 100) > target_accuracy_rate:
    if loss_avg < target_error_rate:
        condition = False
    
        # Save the same samples before and after noise addition
        for i in range(EXAMPLES):
            clean = data[i].cpu().numpy()
            noisy = perturb_audio[i].cpu().detach().numpy()

            # Save clean waveform
            clean_waveform_tensor = torch.tensor(clean).to(device).cpu()
            plt.plot(clean_waveform_tensor.t().numpy())
            plt.savefig(f'sample_clean/clean_plot{i}.png')
            plt.close()
            torchaudio.save(f'sample_clean/clean_{i}.wav', clean_waveform_tensor, SR)
            
            # Save noisy waveform
            noisy_audio_tensor = torch.tensor(noisy).cpu()
            plt.plot(noisy_audio_tensor.t().numpy())
            plt.savefig(f'sample_noise/noise_plot{i}.png')
            plt.close()
            torchaudio.save(f'sample_noise/noise_{i}.wav', noisy_audio_tensor, SR)
## END OF CONDITION LOOP

# Save the first noisy waveform









###############################################
### UPDATE NOISE, SAVE MODEL ###
###############################################

#first_noise = random_noise[0].cpu()
#torchaudio.save('test-testing/ALMOSTEND_first_noisy_sample.wav', first_noise.unsqueeze(0), SR)

# Finale Noise Update to Audio
if torch.is_tensor(random_noise):
    new_random_noise = []
    print("Here in Random Noise is a tensor")
    # Iterate over random_noise
    for idx in range(len(random_noise)):
        sample_noise = random_noise[idx]
        # Get waveform length and create zero mask of same size
        waveform_length = data.shape[2] #CHANGED, was [1]
        mask = np.zeros((waveform_length), np.float32)
        # Get coords of segment and ple noise in location
        start,end = startAndEnd_list[idx]
        mask[start:end] = sample_noise.cpu().numpy()
        #Convert back to tensor and add to list
        new_random_noise.append(torch.from_numpy(mask))
    # Stack list of noises tensors into single tensor
    new_random_noise=torch.stack(new_random_noise)
    random_noise = new_random_noise
#else: random noise isnt a tensor, dont change it
    
## Save the Noise samples
print(f"Final random_noise shape: {random_noise.shape}")
first_noise = random_noise[0].cpu()
torchaudio.save('test-testing/END_first_noisy_sample.wav', first_noise.unsqueeze(0), SR)
torch.save(random_noise, os.path.join(ex_name, 'perturbation.pt'))
print(noise)
print(noise.shape)
print('Noise saved at %s ' % (os.path.join(ex_name, 'perturbation.pt')), flush=True)

print(f"VARIABLES: \n Target_Acc: {target_accuracy_rate}% \n  Number of steps: {train_step}", flush=True)
print(f"Epsilons: {eps_values} \n Step Size: {step_size_factor} \n Eps_cutoff: {eps_cutoff} \n Segment Size: {segment_size}, \n 50TH PERCENTILE", flush=True)



#################################################################
## REFERENCES ##
#################################################################
# 1) https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
# 2) https://github.com/HanxunH/Unlearnable-Examples
#################################################################
