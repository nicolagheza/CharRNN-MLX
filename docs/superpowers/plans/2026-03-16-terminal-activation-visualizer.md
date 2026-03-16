# Terminal Activation Visualizer — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-time terminal visualization that shows input, hidden, and output layer activations during training and inference.

**Architecture:** A separate `visualizer.py` module using `rich` for terminal rendering. The existing `mlx-char-rnn.py` gets refactored into a module with importable functions, then a new `train_viz.py` entry point wires training to the visualizer. The visualizer uses `rich.live.Live` to redraw a dashboard each training step showing: input character, hidden neuron activation heatmap (10x10 grid), and top-k output probabilities.

**Tech Stack:** Python, MLX, Rich (terminal UI), NumPy

---

## File Structure

| File | Responsibility |
|------|---------------|
| `visualizer.py` (create) | Rich-based terminal dashboard — renders activation state |
| `rnn.py` (create) | RNN model + training logic extracted from `mlx-char-rnn.py` as importable functions |
| `train_viz.py` (create) | Entry point that wires training loop to visualizer |
| `mlx-char-rnn.py` (keep) | Original script, unchanged — still works standalone |

---

## Chunk 1: Extract RNN into importable module

### Task 1: Create `rnn.py` with model and training functions

**Files:**
- Create: `rnn.py`

- [ ] **Step 1: Create `rnn.py` with RNN class wrapping model state**

Extract the model into a class so state (weights, hyperparams, vocab mappings) is encapsulated and the forward pass / sampling can be called from external code.

```python
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
```

- [ ] **Step 2: Verify `rnn.py` imports without error**

Run: `cd "/Users/nicolagheza/Developer/Experiments/Character-Level RNN" && python -c "from rnn import CharRNN; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rnn.py
git commit -m "feat: extract RNN model into importable rnn.py module"
```

---

## Chunk 2: Build the terminal visualizer

### Task 2: Create `visualizer.py` with Rich-based dashboard

**Files:**
- Create: `visualizer.py`

- [ ] **Step 1: Create `visualizer.py` with `ActivationVisualizer` class**

The visualizer renders three panels:
1. **Input** — highlights the current input character
2. **Hidden State** — 10x10 colored grid of neuron activations (tanh output, range -1 to 1)
3. **Output** — top 10 predicted characters with probability bars

Uses `rich.live.Live` for real-time updates and `rich.table.Table` + `rich.panel.Panel` for layout.

```python
"""Real-time terminal visualization of RNN activations using Rich."""
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _activation_color(value: float) -> str:
    """Map activation value (-1 to 1) to a color string.
    Negative = blue, zero = black, positive = red."""
    clamped = max(-1.0, min(1.0, value))
    if clamped >= 0:
        intensity = int(clamped * 255)
        return f"rgb({intensity},0,0)"
    else:
        intensity = int(-clamped * 255)
        return f"rgb(0,0,{intensity})"


def _prob_bar(prob: float, width: int = 20) -> str:
    """Create a simple bar string from a probability."""
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


class ActivationVisualizer:
    def __init__(self):
        self.console = Console()
        self.live = None

    def start(self):
        """Start the live display."""
        self.live = Live(console=self.console, refresh_per_second=8, screen=True)
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def build_input_panel(self, input_char: str, target_char: str) -> Panel:
        """Show current input and target characters."""
        text = Text()
        text.append("Input:  ", style="bold")
        text.append(f"'{input_char}'", style="bold green")
        text.append("   Target: ", style="bold")
        text.append(f"'{target_char}'", style="bold yellow")
        return Panel(text, title="Input / Target", border_style="green")

    def build_hidden_panel(self, hidden_state: np.ndarray) -> Panel:
        """Render hidden state as a 10x10 colored grid."""
        h = hidden_state.flatten()
        size = len(h)
        cols = 10
        rows = (size + cols - 1) // cols

        table = Table(show_header=False, show_edge=False, pad_edge=False,
                      padding=(0, 0), show_lines=False)
        for _ in range(cols):
            table.add_column(width=3, justify="center")

        for r in range(rows):
            cells = []
            for c in range(cols):
                idx = r * cols + c
                if idx < size:
                    val = float(h[idx])
                    color = _activation_color(val)
                    cells.append(Text("██", style=color))
                else:
                    cells.append(Text("  "))
            table.add_row(*cells)

        return Panel(table, title=f"Hidden State ({size} neurons)", border_style="blue")

    def build_output_panel(self, probs: np.ndarray, ix_to_char: dict, top_k: int = 10) -> Panel:
        """Show top-k predicted characters with probability bars."""
        p = probs.flatten()
        top_indices = np.argsort(p)[::-1][:top_k]

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Char", width=6, justify="center")
        table.add_column("Prob", width=8, justify="right")
        table.add_column("", width=22)

        for idx in top_indices:
            char = ix_to_char[int(idx)]
            prob = float(p[idx])
            display_char = repr(char) if char in ("\n", "\t", " ") else char
            bar = _prob_bar(prob)
            table.add_row(display_char, f"{prob:.4f}", bar)

        return Panel(table, title="Output Probabilities", border_style="magenta")

    def build_status_panel(self, iteration: int, loss: float, sample_text: str = "") -> Panel:
        """Show training status: iteration, loss, and optional sample text."""
        text = Text()
        text.append(f"Iteration: ", style="bold")
        text.append(f"{iteration}", style="cyan")
        text.append(f"   Loss: ", style="bold")
        text.append(f"{loss:.4f}", style="cyan")
        if sample_text:
            text.append("\n\nSample: ", style="bold")
            text.append(sample_text[:120], style="dim")
        return Panel(text, title="Training Status", border_style="cyan")

    def update(self, *, input_char: str, target_char: str, hidden_state: np.ndarray,
               output_probs: np.ndarray, ix_to_char: dict, iteration: int, loss: float,
               sample_text: str = ""):
        """Redraw the full dashboard."""
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=8),
            Layout(name="main"),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )
        layout["left"].split_column(
            Layout(self.build_input_panel(input_char, target_char), name="input", size=4),
            Layout(self.build_hidden_panel(hidden_state), name="hidden"),
        )
        layout["right"].update(self.build_output_panel(output_probs, ix_to_char))
        layout["status"].update(self.build_status_panel(iteration, loss, sample_text))

        if self.live:
            self.live.update(layout)
```

- [ ] **Step 2: Verify `visualizer.py` imports without error**

Run: `cd "/Users/nicolagheza/Developer/Experiments/Character-Level RNN" && pip install rich && python -c "from visualizer import ActivationVisualizer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add visualizer.py
git commit -m "feat: add Rich-based terminal activation visualizer"
```

---

## Chunk 3: Wire training loop to visualizer

### Task 3: Create `train_viz.py` entry point

**Files:**
- Create: `train_viz.py`

- [ ] **Step 1: Create `train_viz.py` that runs training with live visualization**

```python
"""Train the Character-Level RNN with real-time terminal activation visualization."""
import numpy as np
import mlx.core as mx

from rnn import CharRNN
from visualizer import ActivationVisualizer


def main():
    rnn = CharRNN(data_path="input.txt")
    viz = ActivationVisualizer()

    print(f"Data has {rnn.data_size} characters, {rnn.vocab_size} unique.")
    print("Starting training with visualization... (Ctrl+C to stop)")

    n, p = 0, 0
    smooth_loss = float(-mx.log(1.0 / rnn.vocab_size) * rnn.seq_length)
    hprev = mx.zeros((rnn.hidden_size, 1))
    sample_text = ""

    viz.start()
    try:
        while True:
            # reset if needed
            if p + rnn.seq_length + 1 >= rnn.data_size or n == 0:
                hprev = mx.zeros((rnn.hidden_size, 1))
                p = 0

            inputs = [rnn.char_to_ix[ch] for ch in rnn.data[p : p + rnn.seq_length]]
            targets = [rnn.char_to_ix[ch] for ch in rnn.data[p + 1 : p + rnn.seq_length + 1]]

            # sample every 100 iterations
            if n % 100 == 0:
                sample_ix, _, _ = rnn.sample(hprev, inputs[0], 200)
                sample_text = "".join(rnn.ix_to_char[ix] for ix in sample_ix)

            # forward + backward
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev, activations = rnn.loss_fun(
                inputs, targets, hprev
            )
            smooth_loss = smooth_loss * 0.999 + float(loss) * 0.001

            # update params
            rnn.update_params(dWxh, dWhh, dWhy, dbh, dby)

            # pick the last timestep for visualization
            last_t = len(inputs) - 1
            hidden_np = np.array(activations["hs"][last_t])
            output_np = np.array(activations["ps"][last_t])
            input_char = rnn.ix_to_char[inputs[last_t]]
            target_char = rnn.ix_to_char[targets[last_t]]

            # update visualization
            viz.update(
                input_char=input_char,
                target_char=target_char,
                hidden_state=hidden_np,
                output_probs=output_np,
                ix_to_char=rnn.ix_to_char,
                iteration=n,
                loss=smooth_loss,
                sample_text=sample_text,
            )

            p += rnn.seq_length
            n += 1

    except KeyboardInterrupt:
        viz.stop()
        print(f"\nStopped at iteration {n}, loss: {smooth_loss:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the full pipeline runs**

Run: `cd "/Users/nicolagheza/Developer/Experiments/Character-Level RNN" && timeout 10 python train_viz.py || true`
Expected: The visualization renders for a few seconds then times out. No Python errors.

- [ ] **Step 3: Commit**

```bash
git add train_viz.py
git commit -m "feat: add train_viz.py entry point wiring training to terminal visualizer"
```

---

## Summary

After implementation, run with:
```bash
python train_viz.py
```

The terminal will show a live dashboard with:
- **Top:** Training status (iteration, loss, sample text)
- **Bottom-left:** Input/target characters + 10x10 hidden neuron activation heatmap (blue=negative, black=zero, red=positive)
- **Bottom-right:** Top 10 predicted characters with probability bars

Press `Ctrl+C` to stop.
