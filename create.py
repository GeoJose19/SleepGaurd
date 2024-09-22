import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
source_dir_open = 'mrleyedataset\Open-Eyes'
source_dir_close = 'mrleyedataset\Close-Eyes'
train_dir_open = 'data/train/Open'
train_dir_close = 'data/train/Closed'
valid_dir_open = 'data/valid/Open'
valid_dir_close = 'data/valid/Closed'

# Create directories
os.makedirs(train_dir_open, exist_ok=True)
os.makedirs(train_dir_close, exist_ok=True)
os.makedirs(valid_dir_open, exist_ok=True)
os.makedirs(valid_dir_close, exist_ok=True)

# Get list of files
open_eyes_files = [os.path.join(source_dir_open, f) for f in os.listdir(source_dir_open)]
closed_eyes_files = [os.path.join(source_dir_close, f) for f in os.listdir(source_dir_close)]

# Split files into train and validation sets (80% train, 20% valid)
train_open, valid_open = train_test_split(open_eyes_files, test_size=0.2, random_state=42)
train_close, valid_close = train_test_split(closed_eyes_files, test_size=0.2, random_state=42)

# Move files to respective directories
for file in train_open:
    shutil.copy(file, train_dir_open)

for file in valid_open:
    shutil.copy(file, valid_dir_open)

for file in train_close:
    shutil.copy(file, train_dir_close)

for file in valid_close:
    shutil.copy(file, valid_dir_close)

print("Data organized successfully.")
