from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling, Dropout, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam

class HyperModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1./255, input_shape=input_shape),

            layers.Conv2D(8, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            Flatten(),
            layers.Dense(16, activation='relu'),
            Dropout(0.3),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.0015),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )