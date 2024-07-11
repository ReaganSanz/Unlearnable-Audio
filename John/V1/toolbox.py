import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS
from torch.autograd import Variable
from scipy.signal import butter, lfilter
import numpy as np
import os
import random

SEED = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PerturbationTool:
    def __init__(self, epsilon, step_size, num_steps,seed=0):
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.seed = seed
        np.random.seed(seed)

    def min_min_attack(self, audio_samples, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        device = audio_samples.device
        if random_noise is None:
            random_noise = torch.FloatTensor(*audio_samples.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        perturb_audio = Variable(audio_samples.data + random_noise, requires_grad=True)
        perturb_audio = Variable(torch.clamp(perturb_audio, -1, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_audio], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_audio)
                logits = logits.squeeze(1)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_audio, labels, optimizer)
            perturb_audio.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_audio.grad.data.sign() * (-1)
            perturb_audio = Variable(perturb_audio.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_audio.data - audio_samples.data, -self.epsilon, self.epsilon)
            perturb_audio = Variable(audio_samples.data + eta, requires_grad=True)
            perturb_audio = Variable(torch.clamp(perturb_audio, -1, 1), requires_grad=True)
        return perturb_audio, eta

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
            random.seed(SEED)
            random.shuffle(self._walker)

class PoisonSC(SubsetSC):
    def __init__(self, subset, poison_rate=1.0, perturb_tensor_filepath=None, patch_location='center'):
        super().__init__(subset=subset)
        # load Noise from pertubation.pt, set variables
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.cpu().numpy()
        self.patch_location = patch_location
        self.poison_rate = poison_rate
        self.poisoned_samples = {}
        targets = list(range(len(self)))
        print(f"Total Targets: {len(self)}", flush=True)
        self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
        print(f"Total poison targets: {len(self.poison_samples_idx)}", flush=True)
        
        for idx in self.poison_samples_idx:
         
            noise = self.perturb_tensor[idx % len(self.perturb_tensor)]
            noise, (start,end) = patch_noise_to_sound(noise, waveform_length=16000, segment_location=self.patch_location)
            waveform, sample_rate, label, *_ = self[idx]
            waveform = self._standardize_waveform(waveform)
          
            '''
            if idx == 0:
                print(f"Noise tensor for index {idx}:")
                print(self.perturb_tensor[idx % len(self.perturb_tensor)])
                noise_tensor = torch.tensor(self.perturb_tensor[0]).unsqueeze(0) 
                torchaudio.save("./test-testing/first_noise2.wav", noise_tensor, sample_rate)
                plt.figure()
                plt.plot(self.perturb_tensor[0])
                plt.savefig("./test-testing/first_noise2.png")
                plt.close()
            '''
            num_channels = waveform.shape[0]
            poisoned_waveform = np.zeros_like(waveform.numpy())
            for channel in range(num_channels):
                poisoned_waveform[channel] = np.clip(waveform[channel].numpy() + noise, -1, 1)
            self.poisoned_samples[idx] = (torch.tensor(poisoned_waveform), sample_rate, label)
            '''
            if idx == 0:
                print("Saving first sample...", flush=True)
                torchaudio.save('./test-testing/first_noisy__sample2.wav', torch.tensor(poisoned_waveform), sample_rate)
                plt.figure()
                plt.plot(poisoned_waveform[0]) # was [0]
                plt.savefig('./test-testing/first_noisy_sample2.png')
                plt.close()
            '''
    def __getitem__(self,idx):
        if idx in self.poisoned_samples:
            return self.poisoned_samples[idx]
        else:
            return super().__getitem__(idx)
    
    def _standardize_waveform(self, waveform, target_length=16000):
        waveform_length = waveform.shape[1]
        if waveform_length < target_length:
            padding = target_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform_length > target_length:
            waveform = waveform[:, :target_length]
        return waveform
    
# define Model Architecture
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
    mask = np.zeros(waveform_length, dtype=np.float32)  
    noise_length = noise.shape[0]
    if segment_location == 'center' or (waveform_length == noise_length):
        start = (waveform_length - noise_length) // 2
    elif segment_location == 'random':
        start = np.random.randint(0, waveform_length - noise_length)
    else:
        raise ValueError('Invalid segment location')
    end = start + noise_length
    mask[start:end] = noise
    return mask, (start, end)

# add perturbation noise
def perturb_eval(random_noise, train_loader, model, startAndEnd_list, mask_cord_list):
    print("In Perturb Eval", flush=True)
    loss_meter = AverageMeter()
    err_meter = AverageMeter()
    model = model.to(device)
    idx_v = 0
    # iterate over Data Loader (batches of audio and labels)
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

def train(model, epoch, log_interval, train_loader, transform, optimizer, pbar, pbar_update, losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        #data = transform(data)
        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}") 
        pbar.update(pbar_update)
        losses.append(loss.item())

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def test(model, epoch, total_acc, test_loader, transform, pbar, pbar_update):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        #data = transform(data)
        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        pbar.update(pbar_update)
    acc = (100. * correct / len(test_loader.dataset))
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n")
    total_acc.append(acc)

# helper functions
def label_to_index(word, label_types):
    return torch.tensor(label_types.index(word))

def index_to_label(index, label_types):
    return label_types[index]

def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


