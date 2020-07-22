import pandas as pd

df = pd.read_csv('data/cancer_classification.csv')

df.info()
df.describe().transpose()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='benign_0__mal_1', data=df)

def plot_corr():
    plt.figure(figsize=(12, 8))
    data = df.corr()['benign_0__mal_1'].drop('benign_0__mal_1').sort_values()
    sns.barplot(x=data.index, y=data.values)
    plt.tight_layout()
    plt.xticks(rotation='vertical')

plot_corr()

def corr_heatmap():
    plt.figure(figsize=(12, 12))
    data = df.corr()
    sns.heatmap(data)
    plt.tight_layout()

corr_heatmap()

# Train test split
from sklearn.model_selection import train_test_split

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

X_train, X_test, y_train, y_test  = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=101
)

# Scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example one: with too many epochs and overfitting
def example_one():
    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(
        x=X_train,
        y=y_train,
        epochs=600,
        validation_data=(X_test, y_test),
        verbose=1
    )

    model_loss = pd.DataFrame(model.history.history)

    plt.figure(figsize=(12, 8))
    model_loss.plot()

example_one()


# Example 2: Early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=25
)

def example_two():
    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(
        x=X_train,
        y=y_train,
        epochs=600,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop]
    )

    model_loss = pd.DataFrame(model.history.history)

    plt.figure(figsize=(12, 8))
    model_loss.plot()

example_two()


# Exampler 3: Adding in DropOut Layers
from tensorflow.keras.layers import Dropout

def example_three():
    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(
        x=X_train,
        y=y_train,
        epochs=600,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[early_stop]
    )

    model_loss = pd.DataFrame(model.history.history)

    plt.figure(figsize=(12, 8))
    model_loss.plot()

    return model.predict_classes(X_test)


# Evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = example_three()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
