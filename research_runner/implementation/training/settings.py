"""Worker-injected recipes; there are no legacy deployment defaults."""
def ModernBERTConfig():
    raise RuntimeError("Encoder recipe must be supplied by the runner worker")

def AutoEncoderConfig():
    raise RuntimeError("AE recipe must be supplied by the runner worker")
