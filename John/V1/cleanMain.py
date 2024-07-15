import torch
import torch.optim as optim
import torchaudio
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
import io
import toolbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512            # batch size: increase to speed up job
LOG_INTERVAL = 20           # log interval
EPOCHS = 15                 # number of epochs for training
CUTOFF_FREQUENCY = 4000     # 4 KHz
QUANTIZATION_LEVEL = 256
SR = 16000

# create training and testing split
#train_set = toolbox.SubsetSC("training")
train_set = toolbox.TransformedSC('Training', SR)
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

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
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
 
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

total_acc = []
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(1, EPOCHS + 1):
        print("="*20 + "Training Epoch %d" % (epoch) + "="*20, flush=True)
        toolbox.train(model, epoch, LOG_INTERVAL, train_loader, optimizer, pbar, pbar_update, losses)
        toolbox.test(model, epoch, total_acc, test_loader, pbar, pbar_update)
        scheduler.step()
print (f"Accuracy Plot coords: {total_acc}")
