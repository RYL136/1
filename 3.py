import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

def preprocess(image, label):
    image = tf.image.resize(image, [32, 32])  
    image = tf.cast(image, tf.float32) / 255.0  
    image = (image - 0.5) / 0.5  
    return image, label

def get_datasets():
    train_ds, test_ds = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,  
        shuffle_files=True,
        data_dir="./data"  
    )
    train_ds = (
        train_ds
        .map(preprocess)
        .shuffle(10000)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(preprocess)
        .batch(128)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds

class ImprovedCNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(5, 5),
            padding='SAME',
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=64,
            out_features=128,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=128,
            out_features=256,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            in_features=4096,
            out_features=256,
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            in_features=256,
            out_features=10,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self.conv3(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)
        return x

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)  
    return optax.softmax_cross_entropy(logits, one_hot).mean()

def compute_accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def create_train_state(rng, lr=0.005):
    rngs = nnx.Rngs(rng)
    model = ImprovedCNN(rngs=rngs)
    
    model(jnp.ones((1, 32, 32, 3), dtype=jnp.float32))
    graphdef, params = nnx.split(model)
    warmup, total = 75, 10000
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, lr, warmup),
         optax.cosine_decay_schedule(lr, total - warmup)],
        boundaries=[warmup]
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    
    opt_state = tx.init(params)
    state = nnx.TrainState(
        graphdef=graphdef,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=0
    )
    return state
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        model = nnx.merge(state.graphdef, params)
        logits = model(batch[0])
        loss = cross_entropy_loss(logits, batch[1])
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = compute_accuracy(logits, batch[1])
    return state, loss, acc
@jax.jit
def eval_step(state, batch):
    model = nnx.merge(state.graphdef, state.params)
    logits = model(batch[0])
    loss = cross_entropy_loss(logits, batch[1])
    acc = compute_accuracy(logits, batch[1])
    return loss, acc
def main():
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng)
    train_ds, test_ds = get_datasets()

    for epoch in range(10):
        train_loss, train_acc = [], []
        for batch in tfds.as_numpy(train_ds):
            state, l, a = train_step(state, batch)
            train_loss.append(l)
            train_acc.append(a)
        print(f"Epoch {epoch+1}: Loss={np.mean(train_loss):.4f}, Train Acc={np.mean(train_acc):.4f}")

        test_loss, test_acc = [], []
        for batch in tfds.as_numpy(test_ds):
            l, a = eval_step(state, batch)
            test_loss.append(l)
            test_acc.append(a)
        print(f"Test Acc={np.mean(test_acc):.4f}\n")

    print("Training completed!")

if __name__ == "__main__":
    main()
