import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re
import random
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


set_seed(11) 

def extract_info_from_filename(filename, i):
    """
    Extracts specific parts of the filename based on the provided index `i`.
    The filename format is expected to follow: '[MF](actor_id)(emotion)(scenario_id)V(version).trc'.
    
    Parameters:
        filename (str): The filename to parse.
        i (int): The group index to extract (1 for actor_id, 2 for emotion, etc.).

    Returns:
        str: The extracted part of the filename.
    
    Raises:
        ValueError: If the filename format is not recognized.
    """
    import re
    print(f"Processing Filename: {filename}")
    
    # Match the expected filename format
    match = re.search(r'[MF](\d+)([A-Za-z]{1,2})(\d+)V(\d+)\.trc', filename)
    if match:
        extracted = match.group(i)  # Extract the specified group
        print(f"Extracted Group {i}: {extracted}")
        return extracted.lower()  # Convert to lowercase for consistency
    else:
        raise ValueError(f"Filename format not recognized: {filename}")



def load_trc_files_from_directory(directory_path):
    """
    Loads all .trc files from the specified directory.
    """
    trc_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.trc')]
    return trc_files

# Step 2: Define Data Cleaning and Preprocessing
def clean_and_parse(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    headers = lines[3].strip().split('\t')[2:]
    data = [line.strip().split('\t') for line in lines[5:]]
    
    clean_data = []
    for row in data:
        try:
            clean_row = [float(x) if x != '' else np.nan for x in row]
            clean_data.append(clean_row)
        except ValueError:
            continue

    data_array = np.array(clean_data)
    data_array = data_array[~np.isnan(data_array).any(axis=1)]  # 移除含 NaN 的行
    return headers, data_array


def prepare_stgcn_input(data_array, num_nodes=21):
    """
    Prepares motion capture data for ST-GCN input format.
    """
    num_frames, num_columns = data_array.shape
    num_markers = num_columns // 3  # Each marker has X, Y, Z coordinates

    # Validate the number of markers matches the expected nodes
    if np.isnan(data_array).any() or np.isinf(data_array).any():
        raise ValueError("Data contains NaN or Inf before reshaping.")

    # Reshape data into (T, V, C) where T = frames, V = nodes, C = coordinates (X, Y, Z)
    reshaped_data = data_array[:, :num_nodes * 3].reshape(num_frames, num_nodes, 3)

    # Transpose to match ST-GCN input format: (C, T, V, M)
    # Assume a single person (M=1) in the scene
    stgcn_input = np.transpose(reshaped_data, (2, 0, 1))  # (C=3, T, V)
    return np.expand_dims(stgcn_input, axis=-1)  # Add M=1 dimension

def preprocess_frames(data_array, target_frames=96):
    """
    Preprocess the frames to ensure they are exactly `target_frames` long.
    - If frames are 0, the file is skipped.
    - If frames are less than `target_frames`, pad with zeros.
    - If frames are greater than `target_frames`, truncate.
    """
    num_frames = data_array.shape[0]
    if num_frames == 0:
        return None  # Skip files with 0 frames

    # If frames are less than target, pad with zeros
    if num_frames < target_frames:
        padding = np.zeros((target_frames - num_frames, data_array.shape[1]))
        data_array = np.vstack((data_array, padding))

    # If frames are more than target, truncate
    if num_frames > target_frames:
        data_array = data_array[:target_frames, :]

    return data_array

class PoseDatasetWithLabelsFromDirectory(Dataset):
    def __init__(self, directory_path, target_frames=96):
        self.data_list = []
        self.labels = []
        file_paths = load_trc_files_from_directory(directory_path)
        
        for file_path in file_paths:
            label = extract_info_from_filename(file_path, 2)
            try:
                _, data_array = clean_and_parse(file_path)
                data_array = preprocess_frames(data_array, target_frames)
                if data_array is not None:
                    stgcn_input = prepare_stgcn_input(data_array)
                    if not np.isnan(stgcn_input).any() and not np.isinf(stgcn_input).any():
                        self.data_list.append(stgcn_input)
                        self.labels.append(label)
                    else:
                        print(f"Invalid ST-GCN input for file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return (torch.tensor(self.data_list[idx], dtype=torch.float32),
                self.labels[idx])
# Step 4: Define ST-GCN Model
class STGCN(nn.Module):
    def __init__(self, num_classes=7, num_nodes=21, input_channels=3):
        super(STGCN, self).__init__()
        # Spatial-temporal convolution layers
        self.spatial_conv = nn.Conv2d(input_channels, 64, kernel_size=(1, num_nodes))
        self.temporal_conv = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.spatial_conv(x)  # Spatial convolution
        x = torch.relu(x)
        x = self.temporal_conv(x)  # Temporal convolution
        x = torch.relu(x)
        x = torch.mean(x, dim=(-1, -2))  # Global average pooling
        x = self.fc(x)  # Fully connected layer for classification
        return x

# Step 5: Integrate Data, Model, and Training
# Test with padding/truncating
file_paths = 'trc_data/trc_data'
dataset_with_labels = PoseDatasetWithLabelsFromDirectory(file_paths)

train_losses = []
train_accuracies = []
test_accuracies = []

from torch.utils.data import random_split


train_size = int(0.8 * len(dataset_with_labels))
test_size = len(dataset_with_labels) - train_size
train_dataset, test_dataset = random_split(dataset_with_labels, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Debug dataset contents
for i in range(len(dataset_with_labels)):
    inputs, label = dataset_with_labels[i]
    print(f"Data Shape: {inputs.shape}, Label: {label}")
dataset_with_labels = PoseDatasetWithLabelsFromDirectory(file_paths)
dataloader = DataLoader(dataset_with_labels, batch_size=2, shuffle=True)

# Initialize model, loss, and optimizer
model = STGCN(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


label_mapping = {
    'h': 0, 'n': 1, 'su': 2, 'd': 3, 'f': 4, 'a': 5, 'sa': 6
}


best_accuracy = 0.0  # To store the best validation accuracy
best_model_path = "best_model.pth"  # File path to save the best model

for epoch in range(100): 

    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for inputs, targets in train_dataloader:

        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("Inputs contain NaN or Inf. Skipping batch.")
            continue

        optimizer.zero_grad()
        outputs = model(inputs.squeeze(-1))  

        try:
            label_indices = [label_mapping[label] for label in targets]
        except KeyError as e:
            print(f"Label {e} not found in mapping. Skipping batch.")
            continue

     
        loss = criterion(outputs, torch.tensor(label_indices, dtype=torch.long))
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()


        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == torch.tensor(label_indices)).sum().item()
        total_predictions += len(label_indices)

            # Save the best model


    avg_loss = total_loss / len(train_dataloader)
    train_accuracy = (correct_predictions / total_predictions) * 100
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Training Accuracy = {train_accuracy:.2f}%")

    


    model.eval()
    test_correct_predictions = 0
    test_total_predictions = 0

    with torch.no_grad():  
        for inputs, targets in test_dataloader:
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Skipping batch due to invalid data.")
                continue

            outputs = model(inputs.squeeze(-1)) 

            # Map labels to indexes
            try:
                label_indices = [label_mapping[label] for label in targets]
            except KeyError as e:
                print(f"Label {e} not found in mapping. Skipping batch.")
                continue

            predictions = torch.argmax(outputs, dim=1)
            test_correct_predictions += (predictions == torch.tensor(label_indices)).sum().item()
            test_total_predictions += len(label_indices)

    # Test set accuracy
    test_accuracy = (test_correct_predictions / test_total_predictions) * 100
    test_accuracies.append(test_accuracy)
    print(f"Epoch {epoch+1}: Test Accuracy = {test_accuracy:.2f}%")

        # Save the best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch+1} with validation accuracy {test_accuracy:.2f}%")



epochs = list(range(1, len(train_losses) + 1))

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(epochs, test_accuracies, label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.show()