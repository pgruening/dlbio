import numpy as np 

images = np.load('../Data/data_kaggle/test.npy', encoding='bytes')

for i,img in enumerate(images):

    image = img[1]
    label = np.zeros([image.shape[0],image.shape[1]])
    images[i].append(label)

np.save('../Data/data_kaggle/test_with_labels.npy', images)