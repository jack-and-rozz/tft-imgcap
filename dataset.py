
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def read_data(data_dir, classes, batch_size, img_height, img_width, shuffle=False):
    image_generator = ImageDataGenerator(rescale=1./255) 
    data_gen = image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=data_dir,
        shuffle=shuffle,
        target_size=(img_height, img_width),
        classes=classes,
        class_mode='categorical')
    return data_gen
