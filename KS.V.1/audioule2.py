#imports
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import logging
import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm
# 
#no NVIDIA GPU compatability is available for my device, so this runs on a CPU. much slower runtime
from torchaudio.datasets import SPEECHCOMMANDS
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training variables (batch_size defined after collate function) 
epoch_num = 4
error_rate = 0.5 
accuracy_rate = 85.0 
epsilon = 0.005
step_num = 0.0001/15
train_step = 15
file_save = "Speech Model Testing"

#batch size defined later (256)


class SubsetSC(SPEECHCOMMANDS):  
    #init subset as optional parameter & parent class w/ path and download parameters
    def __init__(self, subset: str = None): 
        super().__init__("./", download=True) 

        #load list of file paths: construct full file path, read it and return list of normal file paths 
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            filepath = filepath.replace("\\", "/") #replacing backslashes. issue with windows
           # print(f"Loading list from: {filepath}") 
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())).replace("\\", "/") for line in fileobj]

        #load validation, testing, or training file paths and set walker attribute depending on how subset parameter is classified 
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")

        #exclude validation & testing to ensure model is trained on separate dataset for better model accuracy
        elif subset == "training": 
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

#create training and tesitng split of data (no validation used)
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

#show waveform data
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

#find list of labels available in data set 
labels_set = sorted(list(set(datapoint[2] for datapoint in train_set)))
new_sample_rate = 8000 
transform=torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
transformed = transform(waveform)

#assign unique index based on position in the list to simplify processing
def label_to_index(word): 
    #return position of word in labels list
    return torch.tensor(labels_set.index(word)) 
 
def index_to_label(index): 
    #return word corresponding to index in labels (inverse of label_to_index)
    return label_set[index] 

"""
#if encoding functions don't work, debug w/this
word_start = "yes"
index = label_to_index(word_start)
word_recovered = index_to_label(index)
print(word_start, "-->", index, "-->", word_recovered)
"""

def pad_sequence(batch): 
    #make tensors in batch the same length by padding w/zeroes
    batch = [item.t() for item in batch] 
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.) 
    return batch.permute(0, 2, 1) 
"""
def pad_sequence(batch):
    tensors = [item[0].unsqueeze() for item in batch]
    targets = torch.tensor([item[2] for item in batch if len(item) > 2])
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value =0)
    return tensors, targets
"""

def collate_fn(batch): 
    #gather in lists to encode labels as indices
    tensors, targets = [], [] 
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
        
    #group list of tensors into batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

#create dataloader for training dataset with specified parameters
train_loader = torch.utils.data.DataLoader(
    train_set,                   #dataset to load
    batch_size = batch_size,    #of samples per batch
    shuffle = True,             #shuffle data at every epoch/training cycle
    collate_fn = collate_fn,    #merge list of samples into batch 
    num_workers = num_workers,  #number of subprocesses to use for data loading (how many subprocesses to load data in parallel)
    pin_memory = pin_memory,    
)

#create dataloader for TESTING data with specified parameters
test_loader = torch.utils.data.DataLoader(
    test_set,                  
    batch_size=batch_size,
    shuffle=False,              
    drop_last=False,           
    collate_fn = collate_fn,    
    num_workers = num_workers,
    pin_memory = pin_memory,
)

#convolutional neural network to process raw audio data 
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



model = M5(n_input=transformed.shape[0], n_output=len(labels_set))
model.to(device)

print(model)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

criteron = nn.CrossEntropyLoss()
noise_shape= [len(train_set), 16000]
noise_rand = torch.zeros(noise_shape)
print("Data setup success! Now printing...")
#######


### Adding adversarial noise 
def add_perturbation(model, noise_rand, train_loader, begin_end, mask_1): 
    #track loss
    idx_1 = 0 
    loss_calc = AverageMeter()
    error_calc = AverageMeter()
    model.eval()
    model = model.to(device)
    begin_end = []
    mask_1 = []
    correct_calc = 0
    
    
    with torch.no_grad(): 
        for batch_idx, (data, target) in enumerate(train_loader): 
            data, target = data.to(device), target.to(device)
            data = data.squeeze(1)
            idx_2 = data.size(2)
            mask = torch.ones_like(data) 
            mask[:, :, idx_1:idx_2] = 0

            begin_end.append((idx_1, idx_2))
            mask_1.append(mask) 
            mask_data = data * mask

            #generate adversarial data + forward pass
            #add_adv, _ = PerturbationTool.min_min_optimize(mask_data, target, model, None, criterion)
            output = model(add_adv)
            loss = criterion(output, target)
            loss_calc += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            #update error calcualtions 
            error_tally = target.size(0) - pred.eq(target.view_as(pred)).sum().item
            error_calc.update(incorrect, data.size(0))
            loss_avg = loss_calc / len(data_loader)
    return avg_loss, accuracy, begin_end, mask_1

print("Perturbation function ran successfully! Moving through PerturbationTool class....")

#Optimization 
class PerturbationTool(): 
#deterimne step size & iterations
   def __init__(self, seed, epsilon, step_size, num_steps):
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    
    #min-min optimization w/PGD added
    def min_min_optimize(self, audio_data, labels, model, optimizer, criterion):
        noise_rand = torch.FloatTensor(*audio_data.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        audio_perturb = Variable(audio_data.data + noise_rand, requires_grad = True)
        audio_perturb = Variable(torch.clamp(audio_perturb, 0, 1), requires_grad=True)
        eta = noise_rand
        print("Now using optimzier on perturbed audio only")
        for _ in range(self.num_steps): 
                optimizer = torch.optim.SGD([perturb_audio], lr=1e-3)
                opt.zero_grad()
                model.zero(grad)
                mod_predict = model(audio_perturb)
                loss = criterion(mod_predict, labels_set) #calc loss
                
                audio_perturb.retain_grad()
                loss.backward()
                
            #adjust perturbation w/gradients, then update audio
        eta = self.step_size * perturb_audio.grad.data.sign() * (-1)
        audio_perturb = Variable(audio_perturb.data + eta, requires_grad = True)
        eta = torch.clamp(audio_perturb - audio_data.data, -self.epsilon, self.epsilon)
        audio_perturb = Variable(audio_data.data + eta, requires_grad = True)
        audio_perturb = Variable(torch.clamp(audio_perturb, 0, 1), requires_grad=True)

            #final perturbed audio(hopefully)
        return perturb_audio, eta


class AverageMeter:
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.avg = 0
        self.count = 0
        self.sum =0 
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    #print to check 
    #print(f'Average Loss: {loss_calc.avg}')


#apply noise to loudest part of waveform 
#grab batch size/audio length, duplicate org audio to apply noise
def noise_application(audio_data, noise_level, random_apply = False): 
    batch_size, _, length = audio_data.clone()
    mask = torch.zeros_like(audio_data) 
    noise_data = audio_data.clone()
    
    #iterate over samples
    for i in range(batch_size): 
        loudest = torch.abs(audio_data[i])
        sorted_index = torch.argsort(loudest, descending = True)
        modded_samples = int(length * noise_level)
        
        #apply noise to random section: otherwise, apply it to loudest parts of waveform
        if random_apply: 
            idx_1 = np.random.randint(0, length - modded_samples)
            idx_2 = idx_1 + modded_samples
        else: 
            index_mod = sorted_index[:modded_samples]
            idx_1 = index_mod[0].item()
            idx_2 = index_mod[-1].item()
        
        idx_1 = max(0, idx_1)
        idx_2 = min(length, idx_2) #end index is not > audio length
        
        # random noise generation 
        print("random noise generating...") #for debugging: if code doesn't reach this area, recheck
        noise = torch.FloatTensor(idx_2 = idx_1).uniform_(-1, 1).to(audio_data.device)
        noise_data[i, :, idx_1:idx_2] += noise
        mask[i, :, idx_1:idx_2] = 1
 
    print("noise application complete. moving to training...") #another print for debug
    return noise_data, mask  

#Model training based on min-min & PGD attack 
model = M5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
noise_level = 0.1 #10% of audiio samples: ensure enough noise is added to test model robustness

train_loss = AverageMeter()
train_error = AverageMeter()
clean_waveforms = []

model.train()
#target CPU device (will run slower), apply loudest parts of data, then zero the gradients
for epoch in range(4):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): 
        try:
            print(f'Epoch {epoch + 1}/{epoch_num}, Batch {batch_idx +1}')
            print(f'Data shape: {data.shape}')
            print(f'Target shape: {target.shape}')

            data, target = data.to(device), target.to(device)
            noise_data, mask = noise_application(data, noise_level=noise_level, random_apply=True)
            #pass, then compute loss, then optimize 
            output = model(noise_data)
            loss = criterion(output, target)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #compute min min optimization 
            noise_data, _ = PerturbationTool.min_min_optimize(noise_data, target, criteron, model)
            output = model(noise_data)
            loss = criterion(output, target)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update calculations (not working: LR & ER is still 0.00)
            train_loss.update(loss.item(), data.size(0))
            class_pred = output.argmax(dim=1, keepdim=True)
            correct_rate = pred.eq(target.view_as(pred)).sum().item()
            error_rate = 1 - correct_rate / data.size(0)

            train_error.update(error_rate, data.size(0))
            #clean_waveforms.append(data.cpu().numpy)
            if batch_idx % 10 == 0: 
                (f'End of Epoch {epoch + 1}/{4}, Average Training Loss: {train_loss.avg:.4f}, Error Rate: {train_error.avg:.4f}')

            #check for error_rate below threshold AND accuracy above threshold
            if error_rate < error_threshold and accuracy_rate > accuracy_threshold: 
                torch.save(data, 'clean_data.pth')
                torch.save(noise_data, 'noisy_data.pth')
         
        except Exception as e: 
            ValueError("ERROR PROCESSING BATCH {batch_idx}: {e}") #debug
        #print results
        print(f'Epoch {epoch + 1}/{epoch_num}, Average Training Loss: {train_loss.avg:.3f}, Error Rate: {train_error.avg:.3f}') # might be cauising calc error?


# Plot training loss versus the number of iterations
"""
plt.plot(loss); 
plt.title("training loss"); 
plt.xlabel("Iterations"); 
plt.ylabel("Loss"); 
plt.show()
"""



"""
#min-min optimization: minimize training & validation errors
def min_min_optimize(model, optimizer, train_loader, noise_rand, data, labels, epochs = 5):

    #inner loop
        for epoch in range(epochs): 
            model.train()
            train_loss_calc = AverageMeter()

            
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                #add noise, then zero parameter gradients
                perturb_data = add_perturbation(model, data, target, epsilon, alpha, step_num, device=device)
                optimizer.zero_grad()

                #forward pass w/ noisy data
                outputs = model(perturb_data)
                loss = criteron(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss_calc.update(loss.item(), inputs.size(0))
                
                #show stats for every 100 batches
                if i % 100 == 99: 
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Average Training Loss: {train_loss_meter.avg:.3f}')

            model.eval()
            val_loss_calc = AverageMeter()
            with torch.no_grad(): 
                for data, target in val_loader: 
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    val_loss_calc.update(loss.item(), data.size(0))

            #adjust learning rate, then reprint
            scheduler.step(val_loss_calc.avg)
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss_meter.avg:.4f}')

     

     ##KEEP TRACK OF MODEL 
class AverageMeter: 
    def __init__(self):
        self.reset()

    def reset(self): 
        self.sum = 0
        self.avg = 0 
        self.count = 0

    def update(self, val, n=1): 
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


# ADD NOISE TO LARGEST PART OF AUDIO WITHIN BOUNDS

#input noise to loudest parts of audio within bounds
def amplify_noise(noise, waveform_length = 16000, segment_location='center', device='cpu'): 
    def center_noise(waveform, noise): 
        mask = torch.zeros_like(waveform)
        start_pos = (waveform_length - noise.size(1)) // 2
        end_pos = start_pos + noise.size(1)

        #check bounds and apply noise to waveforms center
        if start_pos >= 0 and end_pos <= waveform_length:
            mask[:, start_pos:end_pos] = noise
        else: 
            #debug
            raise ValueError("ERROR: Recheck amplify_noise, segment is out of bounds")
        return waveform + mask

    #go through batches, generate noise for each sound
    for data in train_loader: 
        inputs, labels = data
        
        for i in range(inputs.size(0)): 
            inputs[i] = amplify_noise(inputs[i], noise)
    return inputs





#TRAINING THE MODEL !!

#training/testing network (revised from tutorial to include min-min optimization) 
print("Starting min-min training...", flush = True)
epochs = 5
condition = True
for epoch in range(epochs): 
    model.train()

for inputs, labels in train_loader: 
    inputs, labels = inputs.to(device), labels.to(device)

while condition: 
    for batch in train_loader: 
        inputs, labels = inputs.to(device), labels.to(device) 
        perturb_input = patch_noise(inputs) 
        optimizer.zero_grad()
        outputs = model(perturb_input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()
    tot = 0
    correct = 0 
    with torch.no_grad(): 
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / tot
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.3f}')
    
    #stop training if accuracy exceeds threshold
    if accuracy > threshold: 
        condition = False
        break
        #return model 


#training/testing network 
def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

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
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())

#testing networks accuracy, run inference on test dataset
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

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()



"""
"""
References: 
1. https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
2. https://github.com/HanxunH/Unlearnable-Examples
3. https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html

meeting notes: 
- sample wise generation sine function determines direction of your step, alpha determines step size
- gradually reduce loss function 
- use PGD attack 
producing minimizing noise for image
- under while condition, it opimzes for outer loop 
   - perturbatio nover entire dataset = inner optimization (dont update model parameters here) 
   - inner optimization = 1 epoch, outer =10 batches to increase unlearnability. delta makes data unlearnable (inner optimizatio is more important, bc its what makes it unlearnable) 
   - outer optimization makes data unlearnable REGARDLESS of training status
   - perform 20 iterations for pgd attack 
"""
