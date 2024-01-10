import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the text
with open('presentation.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text)) # Get all unique characters
vocab_size = len(vocab) # Number of unique characters
vocab2index = {c: i for i, c in enumerate(vocab)} # Character to index mapping
index2vocab = {i: c for i, c in enumerate(vocab)}
context_size = 8 # Number of characters to consider as context
batch_size = 512 # Number of samples per batch
epoch_size = 100 # Number of times to iterate over the entire dataset

N = 512 # Number of samples in the signal
t = torch.linspace(0, 1, N, requires_grad=False) # Time axis for the signal

def context_to_signal(context):
    signal = torch.zeros(N)
    for i, item in enumerate(context):
            signal += torch.sign(torch.sin(2 * torch.pi * item * (t + i/N)))
    return signal / (torch.max(torch.abs(signal)) + 1e-12)

def sample_from_output(output, temperature=1.0):
    if temperature < 1e-3:
        return torch.argmax(output).item()

    # Adjust output with temperature
    output = output / temperature
    # Compute probabilities using softmax on the correct dimension
    probabilities = F.softmax(output, dim=0)  # Use dim=0 for 1D tensor

    # Sample from the probability distribution
    return torch.multinomial(probabilities, num_samples=1).item()


# Function to generate the dataset
def generate_dataset(text, context_size):
    X = []
    Y = []
    for i in range(len(text)):
        if i < context_size:
            X.append(text[:i])
        else:
            X.append(text[i-context_size:i])
        Y.append(text[i])
    return X, Y

# Generate the dataset
X, Y = generate_dataset(text, context_size)


print(f"Number of samples: {len(X)}, {len(Y)}")

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __getitem__(self, index):
        signal = context_to_signal([vocab2index[c] for c in self.X[index]])
        return signal, torch.tensor([vocab2index[self.Y[index]]], dtype=torch.long)
    
    def __len__(self):
        return len(self.X)


dataset = TextDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class TwoEyesModel(nn.Module):
    def __init__(self, N, eye_features, dropout_rate=0.2):
        super(TwoEyesModel, self).__init__()
        
        self.EYE1 = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(N, eye_features)
        )
        
        self.EYE2 = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(N, eye_features)
        )
        
        self.attention = nn.Linear(eye_features, 1)
        

        self.COMBINE = nn.Sequential(
            nn.Linear(2 * eye_features, N),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(N, N)
        )
        

        self.DECIDE = nn.Sequential(
            nn.Linear(N, N),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(N, vocab_size)
        )

    def forward(self, input_signal):
        eye1 = self.EYE1(input_signal)
        eye2 = self.EYE2(input_signal)
        # Compute attention weights
        attention_weights = F.softmax(
            torch.cat((
                self.attention(eye1),
                self.attention(eye2)
            ), dim=1)
        , dim=1)
        # Apply attention weights
        eyes_signal = torch.cat((attention_weights[:, 0:1] * eye1,  attention_weights[:, 1:2] * eye2), dim=1)
        
        combined = self.COMBINE(eyes_signal)
        
        # Final dense layer
        output = self.DECIDE(input_signal + combined)
        return output


model = TwoEyesModel(N, eye_features=32, dropout_rate=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.to(device)

for epoch in range(epoch_size):
    total_loss = 0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

with torch.no_grad():
    model.eval()
    while True:
        sentence = input("Enter a sentence: ")
        for i in range(500):
            context = sentence[-context_size:]
            encoded = [vocab2index[c] for c in context]
            signal = context_to_signal(encoded)
            output = model(signal.unsqueeze(0))  # Add batch dimension
            pred = sample_from_output(output[0], 0.9)  # Remove batch dimension and sample
            sentence += index2vocab[pred]
        print(sentence)
        print()
