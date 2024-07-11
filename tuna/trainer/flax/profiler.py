import jax

def print_memory_usage():
    for i, dev in enumerate(jax.devices()):
        stats = dev.memory_stats()
        total = stats['bytes_limit']
        used = stats['peak_bytes_in_use']
        print(f"Device {i}:", f"Usage: {used/1e9:.2f} GB / {total/1e9:.2f} GB")