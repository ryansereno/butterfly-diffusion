import pyarrow.parquet as pq
import torch
import os
import torchvision 

# Read the .parquet file
dataset = pq.ParquetDataset('training-data.parquet')
table = pq.read_table(dataset)

# Convert the data to a Pytorch tensor
data = torch.from_numpy(table.to_pydict()['image_column'])

# Create a directory to save the images
if not os.path.exists('training-data/'):
    os.makedirs('training-data/')

# Save the images to the directory
for i, img in enumerate(data):
    torchvision.datasets.ImageFolder.save(img, 'training-data/image_{}.jpg'.format(i))
