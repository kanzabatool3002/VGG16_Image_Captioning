# Image Captioning using VGG16

## Overview
This project implements an image captioning model using the VGG16 deep learning architecture as the feature extractor and an LSTM-based decoder for caption generation. The model achieves an accuracy of **80.43%**.

## Dataset
The dataset consists of images and their corresponding captions stored in a text file.
- Images are located in the `Images` folder.
- Captions are stored in `captions.txt` in a CSV format where each row contains an image filename and its associated captions.
- The dataset is stored in Google Drive and accessed using Google Colab.

## Model Architecture
The model consists of two main components:
1. **Feature Extractor (VGG16):**
   - Pretrained VGG16 is used to extract deep visual features from images.
   - The last fully connected layer (before classification) is used as the image representation.

2. **Text Processing (LSTM Decoder):**
   - Captions are tokenized and converted into sequences.
   - A vocabulary is built, and sequences are padded to a fixed length.
   - An embedding layer maps words to dense vectors.
   - An LSTM layer processes the sequences and generates captions.

## Installation
To run the project, install the required dependencies:

```bash
pip install tensorflow numpy tqdm
```

## Running the Model
1. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Extract Features using VGG16:**
   ```python
   model = VGG16()
   model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
   ```
3. **Process Captions:**
   - Clean text data.
   - Tokenize captions.
   - Pad sequences.
   
4. **Train the Model:**
   ```python
   epochs = 15
   batch_size = 64
   steps = len(train) // batch_size
   
   for i in range(epochs):
       generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
       model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
   ```
5. **Save the Model:**
   ```python
   model.save('/content/best_model.h5')
   ```

## Results
- The model achieves an accuracy of **80.43%** on the validation set.
- It generates captions for unseen images based on learned patterns.
