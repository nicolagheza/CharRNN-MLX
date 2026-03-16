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
            input_np = np.array(activations["xs"][last_t])
            hidden_np = np.array(activations["hs"][last_t])
            output_np = np.array(activations["ps"][last_t])
            input_char = rnn.ix_to_char[inputs[last_t]]
            target_char = rnn.ix_to_char[targets[last_t]]

            # update visualization
            viz.update(
                input_char=input_char,
                target_char=target_char,
                input_vec=input_np,
                hidden_state=hidden_np,
                output_probs=output_np,
                ix_to_char=rnn.ix_to_char,
                input_idx=inputs[last_t],
                target_idx=targets[last_t],
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
