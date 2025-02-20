import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.hyper_model import HyperModel
from models.model import Model
from config import image_size, train_directory, batch_size
import matplotlib.pyplot as plt
import time
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
    'hyper_model': HyperModel,
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    print("Training Accuracy:", acc)
    print("Validation Accuracy:", val_acc)
    print("Training Loss:", loss)
    print("Validation Loss:", val_loss)

    epochs = range(1, len(acc) + 1)
    
    print("Epochs:", list(epochs))

    plt.figure(figsize=(24, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Save the plot to a file
    plt.savefig('results/training_history.png')
    plt.close()

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    # 
    # Your code should change the number of epochs
    epochs = 20  # Increase the number of epochs
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    
    # Apply data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    train_dataset = datagen.flow_from_directory(
        train_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    name = 'hyper_model'
    model_class = models[name]
    print('* Training {} for {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()
    history = model.train_model(train_dataset, validation_dataset, epochs)
    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    
    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), history)
    print('* Model saved as {}'.format(filename))
    plot_history(history)