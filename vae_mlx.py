import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200

# Custom code import
from seqload import loadSeqs

alphabet = "ABCD"
# or, when working with full sequences
# alphabet = "-ACDEFGHIKLMNPQRSTVWY"
q = len(alphabet)


class OneHotGenerator:
    """
    Data generator, converts sequences to one-hot representation in batches.
    """

    def __init__(self, seqs, batch_size):
        self.seqs = seqs
        self.batch_size = batch_size

    def __len__(self):
        return self.seqs.shape[0] // self.batch_size

    def __getitem__(self, idx):
        N = self.batch_size
        batch = self.seqs[idx * N : idx * N + N]

        L = self.seqs.shape[1]
        one_hot = np.zeros((N, L, q), dtype="float32")
        one_hot[np.arange(N)[:, None], np.arange(L)[None, :], batch] = 1
        one_hot = one_hot.reshape((N, L * q))
        return one_hot, one_hot


"""
TODO: Functions to implement:

def generate(self, N):
    # returns a generator yielding sequences in batches
    # assert(N % self.batch_size == 0)

    print("")
    for n in range(N // self.batch_size):
        print("\rGen {}/{}".format(n * self.batch_size, N), end="")

        z = norm.rvs(0.0, 1.0, size=(self.batch_size, self.latent_dim))
        brnll = self.decode_bernoulli(z)

        c = np.cumsum(brnll, axis=2)
        c = c / c[:, :, -1, None]  # correct for fp error
        r = np.random.rand(self.batch_size, self.L)

        seqs = np.sum(r[:, :, None] > c, axis=2, dtype="u1")
        yield seqs
    print("\rGen {}/{}   ".format(N, N))


def decode_bernoulli(self, z):
    brnll = self.decoder.predict(z)
    brnll = brnll.reshape((z.shape[0], self.L, self.q))
    # clip like in Keras categorical_crossentropy used in vae_loss
    brnll = np.clip(brnll, 1e-7, 1 - 1e-7)
    brnll = brnll / np.sum(brnll, axis=-1, keepdims=True)
    return brnll

"""


class Encoder(nn.Module):
    def __init__(self, L, q, hidden_dim, latent_dim, depth):
        super(Encoder, self).__init__()
        self.depth = depth
        self.elu = nn.ELU()
        self.norm = nn.RMSNorm()
        self.dropout = nn.Dropout(0.3)

        self.enc_in = nn.Linear(L * q, hidden_dim)
        self.enc_mid = nn.Linear(hidden_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.mu = 0
        self.log_var = 0
        self.kl = 0
        # the standard normal variation sampling is currently done
        # in the __call__ function, because we use numpy for convenient sampling.

    def __call__(self, x):
        x = self.elu(x)
        for d in range(self.depth):
            x = self.enc_mid(x)
            x = self.elu(x)
            if d % 2 == 0:
                x = self.norm(x)
        mu = self.enc_mu(x)
        sigma = mx.exp(self.enc_log_var(x))
        s = np.random.standard_normal(x.shape)
        s = mx.array(s)
        z = mu + sigma * s
        self.kl = (sigma**2 + mu**2 - mx.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, L, q, latent_dim, hidden_dim, depth):
        super().__init__()
        self.depth = depth
        self.elu = nn.ELU()
        # sigmoid is used as an mx function: `mx.sigmoid()`
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_mid = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, L * q)

    def __call__(self, z):
        z = self.dec_in(z)
        for _ in range(self.depth):
            z = self.elu(self.dec_mid(z))
        z = mx.sigmoid(z)
        return z


class MSAVAE(nn.Module):
    def __init__(self, L, q, hidden_dim, latent_dim, depth):
        super().__init__()
        self.encoder = Encoder(L, q, hidden_dim, latent_dim, depth)
        self.decoder = Decoder(L, q, hidden_dim, latent_dim, depth)

    def __call__(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def loss(self, x, x_target):
        x_hat = self(x)
        xent = mx.mean(nn.losses.cross_entropy(x_hat, x_target))
        mx.simplify(xent)
        return xent - self.encoder.kl


def iterate_batches(batch_size, X, y):
    """
    Yield one batch of data at a time.
    Can optionally randomly permute the index.

    param:
    @X: train sequences
    @y: target sequences
    """
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(X.shape[0])
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]
        s += batch_size
        if s >= X.shape[0]:
            s = 0


def eval_fn(model, val_seq):
    inputs, targets = map(mx.array, to_samples(context_size, val_seq))
    loss = 0
    for s in range(0, targets.shape[0], batch_size):
        bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
        bx, by = map(mx.array, (bx, by))
        losses = model.loss(bx, by, reduce=False)
        loss += mx.sum(losses).item()
    return loss / len(targets)


def main(args):
    optimizer = optim.AdamW()
    latent_dims = 7
    n_batch = 128
    n_epochs = 16
    seqs = loadSeqs(args.seqs, alpha=alphabet)[0]
    N, L = seqs.shape
    vae = MSAVAE(L, q, hidden_dim=1500, latent_dim=latent_dims, depth=10)
    loss_and_grad_fn = nn.value_and_grad(vae, vae.loss)

    for e in range(n_epochs):
        for X, y in iterate_batches(
            n_batch,
        ):
            loss, grads = loss_and_grad_fn(vae, X, y)

            # Update the optimizer state and model parameters
            # in a single call
            optimizer.update(vae, grads)

            # Force a graph evaluation
            mx.eval(vae.parameters(), optimizer.state)

        accuracy = eval_fn(vae, test_images, test_labels)
        print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}")


if __name__ == "__main__":
    main()
