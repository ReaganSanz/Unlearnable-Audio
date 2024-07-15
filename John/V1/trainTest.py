import torch
import torch.optim as optim
import torchaudio
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
import toolbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512            # batch size: increase to speed up job
SR = 16000                  # sample rate
GAMMA = 0.1                 # gamma
PERTURB_PATH = "experiments/perturbation.pt"
LOG_INTERVAL = 20           # log interval
EPOCHS = 15                 # number of epochs for training
POISON_RATE = 1.0



# create training and testing split
train_set = toolbox.SubsetSC("training")
test_set = toolbox.SubsetSC("testing")

# testing first dataset sample
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]
labels = sorted(list(set(datapoint[2] for datapoint in train_set)))



def collate_fn(batch):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [toolbox.label_to_index(label, labels)]
        
    tensors = toolbox.pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets



if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)



# setup model and necessities
model = toolbox.M5(n_input=waveform.shape[0], n_output=len(labels)).to(device)
model.to(device)
poison_train_set = toolbox.PoisonSC("training", poison_rate=POISON_RATE, perturb_tensor_filepath=PERTURB_PATH)
poison_train_loader = torch.utils.data.DataLoader(poison_train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
 
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=GAMMA)
pbar_update = 1 / (len(poison_train_loader) + len(test_loader))
losses = []

#transform = filter_transform.to(device)
total_acc = []
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(1, EPOCHS + 1):
        print("="*20 + "Training Epoch %d" % (epoch) + "="*20, flush=True)
        toolbox.train(model, epoch, LOG_INTERVAL, poison_train_loader, optimizer, pbar, pbar_update, losses)
        toolbox.test(model, epoch, total_acc, test_loader, pbar, pbar_update)
        scheduler.step()
print (f"Accuracy Plot coords: {total_acc}")
# Let's plot the training loss versus the number of iteration.
#plt.plot(losses)
#plt.title("training loss")
