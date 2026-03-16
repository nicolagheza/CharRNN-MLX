"""Character-level Vanilla RNN model — importable module."""
import numpy as np
import mlx.core as mx


class CharRNN:
    def __init__(self, data_path="input.txt", hidden_size=100, seq_length=25, learning_rate=1e-1):
        # data I/O
        data = open(data_path, "r").read()
        self.chars = list(set(data))
        self.data = data
        self.data_size = len(data)
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # hyperparameters
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # model parameters
        self.Wxh = mx.array(np.random.randn(hidden_size, self.vocab_size) * 0.01)
        self.Whh = mx.array(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.Why = mx.array(np.random.randn(self.vocab_size, hidden_size) * 0.01)
        self.bh = mx.zeros((hidden_size, 1))
        self.by = mx.zeros((self.vocab_size, 1))

        # adagrad memory
        self.mWxh = mx.zeros_like(self.Wxh)
        self.mWhh = mx.zeros_like(self.Whh)
        self.mWhy = mx.zeros_like(self.Why)
        self.mbh = mx.zeros_like(self.bh)
        self.mby = mx.zeros_like(self.by)

    def loss_fun(self, inputs, targets, hprev):
        """Forward + backward pass. Returns loss, grads, last hidden state, and
        per-timestep activations dict: {xs, hs, ps}."""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = hprev
        loss = 0

        for t in range(len(inputs)):
            xs_np = np.zeros((self.vocab_size, 1))
            xs_np[inputs[t]] = 1
            xs[t] = mx.array(xs_np)
            hs[t] = mx.tanh(mx.matmul(self.Wxh, xs[t]) + mx.matmul(self.Whh, hs[t - 1]) + self.bh)
            ys[t] = mx.matmul(self.Why, hs[t]) + self.by
            ps[t] = mx.exp(ys[t]) / mx.sum(mx.exp(ys[t]))
            loss += -mx.log(ps[t][targets[t], 0])

        # backward pass
        dWxh = mx.zeros_like(self.Wxh)
        dWhh = mx.zeros_like(self.Whh)
        dWhy = mx.zeros_like(self.Why)
        dbh = mx.zeros_like(self.bh)
        dby = mx.zeros_like(self.by)
        dhnext = mx.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            dy_np = np.array(ps[t])
            dy_np[targets[t]] -= 1
            dy = mx.array(dy_np)
            dWhy += mx.matmul(dy, hs[t].T)
            dby += dy
            dh = mx.matmul(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += mx.matmul(dhraw, xs[t].T)
            dWhh += mx.matmul(dhraw, hs[t - 1].T)
            dhnext = mx.matmul(self.Whh.T, dhraw)

        dWxh = mx.clip(dWxh, -5, 5)
        dWhh = mx.clip(dWhh, -5, 5)
        dWhy = mx.clip(dWhy, -5, 5)
        dbh = mx.clip(dbh, -5, 5)
        dby = mx.clip(dby, -5, 5)

        activations = {"xs": xs, "hs": hs, "ps": ps}
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], activations

    def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
        """Adagrad parameter update."""
        param_names = ["Wxh", "Whh", "Why", "bh", "by"]
        mem_names = ["mWxh", "mWhh", "mWhy", "mbh", "mby"]
        for pname, mname, dparam in zip(param_names, mem_names, [dWxh, dWhh, dWhy, dbh, dby]):
            mem = getattr(self, mname) + dparam * dparam
            setattr(self, mname, mem)
            param = getattr(self, pname) + (-self.learning_rate * dparam / mx.sqrt(mem + 1e-8))
            setattr(self, pname, param)

    def sample(self, h, seed_ix, n):
        """Sample n characters from the model."""
        x_np = np.zeros((self.vocab_size, 1))
        x_np[seed_ix] = 1
        x = mx.array(x_np)
        ixes = []
        hidden_states = []
        output_probs = []

        for t in range(n):
            h = mx.tanh(mx.matmul(self.Wxh, x) + mx.matmul(self.Whh, h) + self.bh)
            y = mx.matmul(self.Why, h) + self.by
            p = mx.exp(y) / mx.sum(mx.exp(y))

            hidden_states.append(h)
            output_probs.append(p)

            cumsum = mx.cumsum(p, axis=0)
            r = mx.random.uniform(0.0, 1.0)
            ix = int(mx.sum(cumsum < r))

            x_np = np.zeros((self.vocab_size, 1))
            x_np[ix] = 1
            x = mx.array(x_np)
            ixes.append(ix)

        return ixes, hidden_states, output_probs
