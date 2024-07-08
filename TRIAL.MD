### Experiment - 1

- Experiment - I: Dropouts After Conv and FC layers
    - In the first experiment, we will use dropouts both after the convolutional and fully connected layers.
        - Accuracy:  0.5669642686843872
        - Validation Accuracy:  0.5212528109550476

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```

### Experiment - 2

- Experiment - II:
    - Remove the dropouts after the convolutional layers (but retain them in the FC layer). Also, use batch normalization after every convolutional layer.
        - Accuracy:  0.6690848469734192
        - Validation Accuracy:  0.48769575357437134
        - Loss:  1.0325521230697632
        - Validation Loss 2.097219467163086

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```

### Experiment - 3

- Experiment - III:
    - Use batch normalization and dropouts after every convolutional layer. Also, retain the dropouts in the FC layer.
        - Accuracy:  0.5892857313156128
        - Validation Accuracy:  0.5145413875579834
        - Loss:  1.4014623165130615
        - Validation Loss 2.195526123046875

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```

### Experiment - 4

- Experiment - IV:
    - Remove the dropouts after the convolutional layers and use L2 regularization in the FC layer. Retain the dropouts in FC.
        - Accuracy:  0.9112723469734192
        - Validation Accuracy:  0.5973154306411743
        - Loss:  0.44464144110679626
        - Validation Loss 1.6497119665145874

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(512,activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```

### Experiment - 5

- Experiment - V:
    - Dropouts after conv layer, L2 in FC, use BN after convolutional layer
        - Accuracy:  0.6774553656578064
        - Validation Accuracy:  0.536912739276886
        - Loss:  1.1889420747756958
        - Validation Loss 1.942699909210205

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```

### Experiment - 6

- Experiment - VI:
    - Dropouts after conv layer, L2 in FC, use BN after convolutional layer
        - Accuracy:  0.6774553656578064
        - Validation Accuracy:  0.536912739276886
        - Loss:  1.1889420747756958
        - Validation Loss 1.942699909210205

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(layers.Rescaling(1./255, input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),padding = 'same',activation= 'relu'))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(7, 7),padding = 'same',activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(11,11),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3, 3),padding = 'same',activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
```