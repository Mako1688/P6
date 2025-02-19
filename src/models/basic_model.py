from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )