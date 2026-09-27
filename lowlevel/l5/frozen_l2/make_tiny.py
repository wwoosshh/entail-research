"""A small model of a given architecture with random weights, saved with a real tokenizer, so a comparison that
needs the architecture's code path (not its knowledge) can run on one machine. Seeded; no download.
  python lowlevel/l2/make_tiny.py mamba2 <tokenizer dir> <out dir>
"""
import shutil
import sys


def main():
    import torch
    from transformers import AutoTokenizer

    kind, tok_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(tok_dir)
    vocab = ((len(tok) + 63) // 64) * 64
    if kind == "mamba2":
        from transformers import Mamba2Config, Mamba2ForCausalLM

        cfg = Mamba2Config(hidden_size=64, num_hidden_layers=2, state_size=16, num_heads=8, head_dim=16, n_groups=1,
                           expand=2, conv_kernel=4, chunk_size=8, vocab_size=vocab, use_bias=True, use_conv_bias=True)
        model = Mamba2ForCausalLM(cfg)
    else:
        raise SystemExit(f"unknown kind {kind}")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("saved", out, "vocab", vocab, "params", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
