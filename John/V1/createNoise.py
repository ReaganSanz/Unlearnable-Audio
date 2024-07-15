import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
import os
import toolbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256            # batch size (increase to speed up job) as 256
ERROR_RATE = 0.01           # loss threshold
ACCURACY = 99.5             # accuracy threshold increase
EPSILON = 0.01              # epsilon   increase, make sure noise within range
STEP_SIZE = EPSILON/25      # distance of each step (in min-min attack)
STEPS = 20                  # number of train steps the model will do in each epoch (during Min-Min attack) increase
EX_NAME = "experiments"     # folder name to save model to
SR = 16000                  # sample rate
GAMMA = 0.1                 # gamma
EXAMPLES = 3                # number of example audios saved



# create training and testing split
train_set = toolbox.SubsetSC("training")
train_sampler = SubsetRandomSampler(torch.randperm(len(train_set)))


# testing first dataset sample
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
print ("==Test [0]==", flush=True)
print(f"Testset[0]: {train_set[0]}")
print(f" Waveform {waveform} \n Sample Rate: {sample_rate} \n Label: {label} \n Utterance Num: {utterance_number}", flush=True)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

label_types = sorted(list(set(datapoint[2] for datapoint in train_set)))

def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [toolbox.label_to_index(label, label_types)]
    tensors = toolbox.pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets



if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)



# setup model and necessities
model = toolbox.M5(n_input=waveform.shape[0], n_output=len(label_types))
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()
noise_shape = [len(train_set), 16000]
random_noise = torch.zeros(noise_shape)
 


idx_temp = 0
mask_cord_list = []
startAndEnd_list = []
idx = 0

for data, labels in train_loader:
    for i, (datum,label) in enumerate(zip(data,labels)):
        noise = random_noise[idx].numpy() 
        mask_cord, startAndEnd = toolbox.patch_noise_to_sound(noise, waveform_length=datum.shape[1], segment_location='center') 
        startAndEnd_list.append(startAndEnd)
        mask_cord_list.append(mask_cord)
        idx += 1



# train model on perturbations
print("Start Training/Min-min", flush=True)
condition = True
train_idx = 0
data_iter = iter(train_loader)
clean_waveform_list =[]
noisy_waveform_list = []
print('=' * 20 + 'Searching Samplewise Perturbuations' + '=' * 20, flush=True)

while condition:
    for j in tqdm(range(STEPS)):
        try:
            (data,labels) = next(data_iter)
        except: 
            train_idx = 0
            data_iter = iter(train_loader)
            (data,labels) = next(data_iter)

        data,labels = data.to(device), labels.to(device)

        for i, (datum,label) in enumerate(zip(data,labels)):
            sample_noise = random_noise[train_idx]
            mask = np.zeros(datum.shape[1], np.float32)
            start,end = startAndEnd_list[train_idx]
            mask[start:end] = sample_noise.cpu().numpy()
            sample_noise = torch.from_numpy(mask).to(device)
            data[i] = data[i] + sample_noise
            train_idx += 1
           
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        model.zero_grad()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(),labels)
        loss.backward()
        optimizer.step()

    idx = 0
    for i, (data,labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)
        batch_noise, batch_start_idx = [], idx
        for j, datum in enumerate(data):
            sample_noise = random_noise[idx]
            mask = np.zeros(datum.shape[1], np.float32)
            start,end = startAndEnd_list[idx]
            mask[start:end] = sample_noise.cpu().numpy()
            sample_noise = torch.from_numpy(mask).to(device)
        
            datum_cpu = datum.cpu().numpy()
            clean_waveform_list.append(datum_cpu)

            noisy_waveform = datum_cpu + mask
            noisy_waveform = torch.tensor(noisy_waveform).to(device)
            batch_noise.append(noisy_waveform)
            idx += 1

        # eval the model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        ## MIN-MIN Attack
        batch_noise = torch.stack(batch_noise)
        attack = toolbox.PerturbationTool(EPSILON, STEP_SIZE, STEPS)
        perturb_audio, eta = attack.min_min_attack(data, labels, model, optimizer, criterion, random_noise=batch_noise)

        for i, delta in enumerate(eta):
            mask_cord = mask_cord_list[batch_start_idx + i]
            delta_cpu = delta.detach().cpu().numpy()
            random_noise[batch_start_idx + i] = torch.tensor(delta_cpu).to(device)

    loss_avg, error_rate = toolbox.perturb_eval(random_noise, train_loader,model,startAndEnd_list=startAndEnd_list, mask_cord_list=mask_cord_list)
    print('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate * 100), flush=True)

    if loss_avg < ERROR_RATE:
        condition = False
    
        # save the same samples before and after noise addition
        for i in range(EXAMPLES):
            clean = data[i].cpu().numpy()
            noisy = perturb_audio[i].cpu().detach().numpy()

            # save clean waveform
            clean_waveform_tensor = torch.tensor(clean).to(device).cpu()
            plt.plot(clean_waveform_tensor.t().numpy())
            plt.savefig(f'sample_clean/clean_plot{i}.png')
            plt.close()
            torchaudio.save(f'sample_clean/clean_{i}.wav', clean_waveform_tensor, SR)
            
            # save noisy waveform
            noisy_audio_tensor = torch.tensor(noisy).cpu()
            plt.plot(noisy_audio_tensor.t().numpy())
            plt.savefig(f'sample_noise/noise_plot{i}.png')
            plt.close()
            torchaudio.save(f'sample_noise/noise_{i}.wav', noisy_audio_tensor, SR)



# update noise and save model

# finalize noise to audio
if torch.is_tensor(random_noise):
    new_random_noise = []
    for idx in range(len(random_noise)):
        sample_noise = random_noise[idx]
        waveform_length = data.shape[2] #CHANGED, was [1]
        mask = np.zeros((waveform_length), np.float32)
        start,end = startAndEnd_list[idx]
        mask[start:end] = sample_noise.cpu().numpy()
        new_random_noise.append(torch.from_numpy(mask))
    new_random_noise=torch.stack(new_random_noise)
    random_noise = new_random_noise
    
# save the Noise samples
print(f"Final random_noise shape: {random_noise.shape}")
first_noise = random_noise[0].cpu()
torchaudio.save('test-testing/END_first_noisy_sample.wav', first_noise.unsqueeze(0), SR)
torch.save(random_noise, os.path.join(EX_NAME, 'perturbation.pt'))
print(noise)
print(noise.shape)
print('Noise saved at %s ' % (os.path.join(EX_NAME, 'perturbation.pt')), flush=True)
print(f"VARIABLES: \n Target_Acc: {ACCURACY}% \n Epsilon: {EPSILON} \n Step Size: {STEP_SIZE} \n Number of steps: {STEPS}", flush=True)
