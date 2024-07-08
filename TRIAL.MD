### Experiment - 1

- Experiment - I: Dropouts After Conv and FC layers
    - In the first experiment, we will use dropouts both after the convolutional and fully connected layers.
        - Accuracy:  0.4665178656578064
        - Validation Accuracy:  0.33109620213508606
        - Loss:  1.510475993156433
        - Validation Loss 1.8774781227111816

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 2

- Experiment - II:
    - Remove the dropouts after the convolutional layers (but retain them in the FC layer). Also, use batch normalization after every convolutional layer.
        - Accuracy:  0.4174107015132904
        - Validation Accuracy:  0.427293062210083
        - Loss:  1.59054434299469
        - Validation Loss 1.9155768156051636

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 3

- Experiment - III:
    - Use batch normalization and dropouts after every convolutional layer. Also, retain the dropouts in the FC layer.
        - Accuracy:  0.4832589328289032
        - Validation Accuracy:  0.48098433017730713
        - Loss:  1.5458106994628906
        - Validation Loss 1.911045789718628

```shell
]img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 4

- Experiment - IV:
    - Remove the dropouts after the convolutional layers and use L2 regularization in the FC layer. Retain the dropouts in FC.
        - Accuracy:  0.51171875
        - Validation Accuracy:  0.501118540763855
        - Loss:  1.7745400667190552
        - Validation Loss 1.9198698997497559

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 5

- Experiment - V:
    - Dropouts after conv layer, L2 in FC, use BN after convolutional layer
        - Accuracy:  0.5636160969734192
        - Validation Accuracy:  0.5391498804092407
        - Loss:  1.9516723155975342
        - Validation Loss 2.279395341873169

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 6

- Experiment - VI:
    - Add a new convolutional layer to the network. Note that by a 'convolutional layer', the professor is referring to a convolutional unit with two sets of Conv2D layers with 128 filters each (we are abusing the terminology a bit here).
        - Accuracy:  0.5256696343421936
        - Validation Accuracy:  0.48322147130966187
        - Loss:  1.8941553831100464
        - Validation Loss 2.1155521869659424

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))

model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

### Experiment - 7

- Experiment - VII:
    - feature maps.
        - Accuracy:  0.4006696343421936
        - Validation Accuracy:  0.37807607650756836
        - Loss:  3.017854690551758
        - Validation Loss 3.102226972579956

```shell
img_rows, img_cols = 180, 180
input_shape = (img_rows, img_cols, 3)
num_classes = len(class_names)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```