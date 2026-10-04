import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

iris = sns.load_dataset('iris')
df = pd.get_dummies(iris, columns=['species'], dtype=int)

x = df.iloc[:, :4]
y = df.iloc[:, 4:]

model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

esc = EarlyStopping(patience=50, monitor='val_loss')
history = model.fit(x, y, callbacks=[esc],
                    validation_split=0.25, 
                    epochs=10000, verbose=1)

print(f'accuracy : {model.evaluate(x,y)}')

res = pd.DataFrame({'species':y.to_numpy().argmax(axis=1),
                     'pred':model.predict(x).argmax(axis=1)})
print(f'correct:{res[res.species==res.pred].species.count()}, incorrect:{res[res.species!=res.pred].species.count()}')
print(res[res.species != res.pred])