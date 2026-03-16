"""Real-time terminal visualization of RNN activations using Rich.

Draws the actual neural network: input, hidden, and output layers as columns
of neurons (circles) with connections between them. Neurons are colored by
activation intensity.
"""
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def _activation_color(value: float) -> str:
    """Map activation value to a color.
    Negative = blue, zero = dark gray, positive = red/yellow."""
    clamped = max(-1.0, min(1.0, value))
    if clamped >= 0:
        r = int(155 + clamped * 100)
        g = int(clamped * 180)
        return f"rgb({r},{g},0)"
    else:
        intensity = int(-clamped * 255)
        return f"rgb(0,{intensity // 3},{intensity})"


def _prob_color(prob: float) -> str:
    """Map probability (0-1) to green intensity."""
    g = int(80 + prob * 175)
    return f"rgb(0,{g},0)"


# Characters for drawing
NEURON_ON = "●"
NEURON_DIM = "○"
CONN_H = "─"
CONN_DIAG_DOWN = "╲"
CONN_DIAG_UP = "╱"


class ActivationVisualizer:
    def __init__(self, max_input_neurons=15, max_hidden_neurons=20, max_output_neurons=15):
        self.console = Console()
        self.live = None
        self.max_input = max_input_neurons
        self.max_hidden = max_hidden_neurons
        self.max_output = max_output_neurons

    def start(self):
        self.live = Live(console=self.console, refresh_per_second=8, screen=True)
        self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def _subsample(self, values, max_n):
        """Pick top-activated neurons if there are too many to display."""
        if len(values) <= max_n:
            return list(range(len(values))), values
        indices = np.argsort(np.abs(values))[::-1][:max_n]
        indices = np.sort(indices)
        return indices.tolist(), values[indices]

    def build_network_panel(self, input_vec, hidden_state, output_probs,
                            ix_to_char, input_idx, target_idx) -> Panel:
        """Draw the neural network as three columns of neurons with connections."""
        input_flat = input_vec.flatten()
        hidden_flat = hidden_state.flatten()
        output_flat = output_probs.flatten()

        # Subsample layers for display
        in_indices, in_vals = self._subsample(input_flat, self.max_input)
        hid_indices, hid_vals = self._subsample(hidden_flat, self.max_hidden)
        out_top = np.argsort(output_flat)[::-1][:self.max_output]
        out_indices = np.sort(out_top).tolist()
        out_vals = output_flat[out_indices]

        n_in = len(in_vals)
        n_hid = len(hid_vals)
        n_out = len(out_vals)
        n_rows = max(n_in, n_hid, n_out)

        # Column positions (character offsets)
        col_in = 0
        col_conn1 = 8
        col_hid = 18
        col_conn2 = 28
        col_out = 38
        col_label = 42
        total_width = 56

        text = Text()

        # Header
        header = " " * col_in + "Input"
        header += " " * (col_hid - len(header)) + "Hidden"
        header += " " * (col_out - len(header)) + "Output"
        text.append(header + "\n", style="bold dim")

        sizes_line = f" ({len(input_flat)})"
        sizes_line += " " * (col_hid - len(sizes_line)) + f"  ({len(hidden_flat)})"
        sizes_line += " " * (col_out - len(sizes_line)) + f" ({len(output_flat)})"
        text.append(sizes_line + "\n", style="dim")
        text.append("─" * total_width + "\n", style="dim")

        for row in range(n_rows):
            line_parts = []

            # --- Input neuron ---
            if row < n_in:
                idx = in_indices[row]
                val = float(in_vals[row])
                char_label = ix_to_char.get(idx, "?")
                if char_label in ("\n", "\t", " "):
                    char_label = repr(char_label)[1:-1]
                is_active = val > 0.5
                neuron = NEURON_ON if is_active else NEURON_DIM
                color = "bold green" if is_active else "dim"
                label = f"{char_label:>2} "
                text.append(label, style="dim")
                text.append(neuron, style=color)
            else:
                text.append("   " + " ", style="dim")

            # --- Connection input -> hidden ---
            gap1 = col_conn1 - 4
            if row < n_in and row < n_hid:
                conn_style = "dim green" if float(in_vals[row]) > 0.5 else "dim"
                text.append(" " + CONN_H * (gap1 - 1), style=conn_style)
            else:
                text.append(" " * gap1, style="dim")

            # --- Hidden neuron ---
            pad_to_hid = col_hid - col_conn1 - gap1 + 4
            if row < n_hid:
                val = float(hid_vals[row])
                color = _activation_color(val)
                text.append(NEURON_ON, style=color)
            else:
                text.append(" ")

            # --- Connection hidden -> output ---
            gap2 = col_conn2 - col_hid - 1
            if row < n_hid and row < n_out:
                h_val = float(hid_vals[row])
                conn_color = _activation_color(h_val) if abs(h_val) > 0.1 else "dim"
                text.append(CONN_H * gap2, style=conn_color)
            else:
                text.append(" " * gap2)

            # --- Output neuron ---
            pad_to_out = col_out - col_conn2 - gap2
            if row < n_out:
                idx = out_indices[row]
                prob = float(out_vals[row])
                color = _prob_color(prob)
                text.append(NEURON_ON, style=color)
                # Label with character and probability
                char_label = ix_to_char.get(idx, "?")
                if char_label in ("\n", "\t", " "):
                    char_label = repr(char_label)[1:-1]
                bar_len = int(prob * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                is_target = (idx == target_idx)
                label_style = "bold yellow" if is_target else "dim"
                text.append(f" {char_label:>2} ", style=label_style)
                text.append(bar, style=color)
                text.append(f" {prob:.3f}", style="dim")
            else:
                text.append(" ")

            text.append("\n")

            # Add ellipsis indicators
            if row == n_rows - 1:
                ellipsis_line = ""
                if len(input_flat) > self.max_input:
                    ellipsis_line += f"   ⋮ (+{len(input_flat) - self.max_input})"
                else:
                    ellipsis_line += "      "
                ellipsis_line += " " * (col_hid - len(ellipsis_line))
                if len(hidden_flat) > self.max_hidden:
                    ellipsis_line += f"⋮ (+{len(hidden_flat) - self.max_hidden})"
                else:
                    ellipsis_line += "  "
                ellipsis_line += " " * (col_out - len(ellipsis_line))
                if len(output_flat) > self.max_output:
                    ellipsis_line += f"⋮ (+{len(output_flat) - self.max_output})"
                if "⋮" in ellipsis_line:
                    text.append(ellipsis_line + "\n", style="dim")

        # Legend
        text.append("\n")
        text.append("─" * total_width + "\n", style="dim")
        text.append(f"  Input: ", style="bold")
        text.append(NEURON_ON + " active  ", style="bold green")
        text.append(NEURON_DIM + " inactive", style="dim")
        text.append(f"   Hidden: ", style="bold")
        text.append(NEURON_ON, style="rgb(255,180,0)")
        text.append(" +  ", style="dim")
        text.append(NEURON_ON, style="rgb(0,85,255)")
        text.append(" −", style="dim")
        text.append(f"   Output: ", style="bold")
        text.append("█ probability\n", style="rgb(0,200,0)")

        return Panel(text, title="Neural Network", border_style="bright_white")

    def build_status_panel(self, iteration, loss, input_char, target_char,
                           sample_text="") -> Panel:
        text = Text()
        text.append("Iter: ", style="bold")
        text.append(f"{iteration}", style="cyan")
        text.append("   Loss: ", style="bold")
        text.append(f"{loss:.4f}", style="cyan")
        text.append("   Input: ", style="bold")
        text.append(f"'{input_char}'", style="bold green")
        text.append(" → Target: ", style="bold")
        text.append(f"'{target_char}'", style="bold yellow")
        if sample_text:
            text.append("\n\nGenerated: ", style="bold")
            text.append(sample_text[:200].replace("\n", "↵"), style="dim")
        return Panel(text, title="Training Status", border_style="cyan")

    def update(self, *, input_char, target_char, input_vec, hidden_state,
               output_probs, ix_to_char, input_idx, target_idx,
               iteration, loss, sample_text=""):
        """Redraw the full dashboard."""
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=6 if not sample_text else 9),
            Layout(name="network"),
        )
        layout["status"].update(
            self.build_status_panel(iteration, loss, input_char, target_char, sample_text)
        )
        layout["network"].update(
            self.build_network_panel(input_vec, hidden_state, output_probs,
                                     ix_to_char, input_idx, target_idx)
        )
        if self.live:
            self.live.update(layout)
