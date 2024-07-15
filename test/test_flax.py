import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

print("사용 중인 장치:", jax.devices()[0])

# 간단한 MLP 모델 정의
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

# 모델 초기화 함수
def create_train_state(rng, learning_rate):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 784]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

# 손실 함수
@jax.jit
def compute_loss(params, images, labels):
    logits = MLP().apply({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss

# 훈련 스텝 함수
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        return compute_loss(params, images, labels)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 메인 훈련 루프
def train_model():
    key = jax.random.PRNGKey(0)
    state = create_train_state(key, learning_rate=1e-3)

    # 임의의 훈련 데이터 생성
    num_samples = 1000
    images = jax.random.normal(key, (num_samples, 784))
    labels = jax.random.randint(key, (num_samples,), 0, 10)

    for epoch in range(10):
        state, loss = train_step(state, images, labels)
        print(f"Epoch {epoch+1}, Loss: {loss}")

    return state

# 모델 훈련 실행
final_state = train_model()
print("훈련 완료")