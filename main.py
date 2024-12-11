# %% [markdown]
# # TF Keras Model with loss(x, y_true, y_pred)

# %%
from helper import *
import helper

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

sns.set_theme(style="ticks")

penguins_df = sns.load_dataset("penguins")

penguins_df = penguins_df.dropna()

# One-hot encode categorical features
penguins_df = pd.get_dummies(penguins_df, drop_first=True, dtype=float)

# penguins_df.sample(5, random_state=42)

# %%
train, temp = train_test_split(penguins_df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# train.shape, val.shape, test.shape

# %%
def data_gen(df: pd.DataFrame, batch_size: int = 32):
    num_samples: int = len(df)

    # One-hot encode categorical features
    # df = pd.get_dummies(df, drop_first=True, dtype=float)

    while True:  # Infinite loop for continuous data generation
        # Shuffle the DataFrame at the beginning of each epoch
        df = df.sample(frac=1).reset_index(drop=True)

        # Generate batches
        for start_idx in range(0, num_samples - num_samples % batch_size, batch_size):  # Drop the remainder
            batch_df = df.iloc[start_idx:start_idx + batch_size]

            # Collect data
            Xs = []
            ys = []

            for df_index, row in batch_df.iterrows():
                # Convert row to numpy arrays (or other required transformations)
                X_row = row.drop(labels=["body_mass_g"]).values  # extracting feature values
                y_row = row["body_mass_g"]  # extracting target value
                
                Xs.append(X_row)
                ys.append(y_row)

            # Convert to numpy arrays for training
            Xs = np.array(Xs)
            ys = np.array(ys)

            yield Xs, ys


# %%
batch_size = 16  # 32
first_model_layer_shape = (train.shape[1] - 1,)

train_gen = data_gen(train, batch_size=batch_size)
val_gen   = data_gen(val,   batch_size=batch_size)
test_gen  = data_gen(test,  batch_size=batch_size)

# %%
# for i, e in zip(range(5), train_gen):
#    print(e)

# %%
def model_1(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    # x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation=None)(x)
    return tf.keras.Model(inputs, outputs)

# %%
def my_loss_fn(x, y_true, y_pred):
    # Ensure y_true has the same data type as y_pred
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    
    return tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))

# %%
flag: bool = True

opt = Adam(learning_rate=0.001, clipnorm=1.0)

if flag:
    model = model_1(input_shape=first_model_layer_shape)
    model.compile(optimizer=opt, loss="mae", metrics=["mae", "mse", "mape"])
else:
    # NOTE: loss could acces X
    model = XModel(model_1(input_shape=first_model_layer_shape))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss_fn=my_loss_fn
        # (lambda x, y_true, y_pred: ...)  # L2 norm
        )
    # , metrics=["mae", "mape"])


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    steps_per_epoch=50,
    validation_steps=10
)

# import datetime
# model.save(f"{datetime.datetime.now().isoformat().split('.')[0]}-model.keras")

# Evaluation on test set
print(f"{model.evaluate(test_gen, steps=1) = }")  # :.3%

# %%
# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE During Training')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# model.summary()


