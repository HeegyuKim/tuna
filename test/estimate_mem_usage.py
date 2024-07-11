import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax_adam_mini import adam_mini
import optax
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def get_memory_usage():
    jax.debug.print("Memory usage:")
    for i, dev in enumerate(jax.devices()):
        stats = dev.memory_stats()
        total = stats['bytes_limit']
        used = stats['peak_bytes_in_use']
        # "peak_bytes_in_use" or "bytes_in_use"

        jax.debug.print(f"Device {i}: Usage: {used/1e9:.2f} GB / {total/1e9:.2f} GB")
        # print(stats)
    jax.debug.print("--------------------")

# Initialize model and tokenizer
model_name = "gpt2"  # or any other causal language model
model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
# print(model.params[list(model.params.keys())[0]])
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create dummy dataset (replace with your actual dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
max_length = 1024

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

print("Hi")
# Create train state
learning_rate = 2e-5
# optimizer = optax.adamw(learning_rate=learning_rate)
# optimizer = optax.adafactor(learning_rate=learning_rate)
optimizer = adam_mini(
    learning_rate=learning_rate,
    n_embd=model.config.n_embd,
    n_head=model.config.n_head,
    n_query_groups=model.config.n_head,
    weight_decay=0
    )

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optimizer,
    optax.scale(-1)
)
@jax.jit
def create_train_state(params):
    return train_state.TrainState.create(
        apply_fn=model.__call__,
        params=params,
        tx=optimizer,
    )

state = create_train_state(model.params)
get_memory_usage()

# Training step function
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], params=params)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = batch["input_ids"][..., 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1)
        ).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    get_memory_usage()
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
batch_size = 4  # Reduced batch size due to potentially larger model
num_epochs = 3

for epoch in range(num_epochs):
    for i in range(0, len(tokenized_dataset), batch_size):
        batch = tokenized_dataset[i:i+batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        
        state, loss = train_step(state, batch)
        
        # Print memory usage after each step
        if i % 10 == 0:  # Print every 10 steps to reduce output
            print(f"Epoch {epoch+1}, Step {i//batch_size+1}, Loss: {loss:.4f}")

# Final memory usage
print("Final memory usage:")
print(get_memory_usage())