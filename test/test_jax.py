import jax
import jax.numpy as jnp

# TPU 사용 여부 확인
print("사용 중인 장치:", jax.devices()[0])

# 간단한 행렬 곱셈 함수 정의
def matrix_multiply(a, b):
    return jnp.dot(a, b)

# 랜덤 행렬 생성
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 1000))
y = jax.random.normal(key, (1000, 1000))

# 함수 JIT 컴파일 및 실행
matrix_multiply_jit = jax.jit(matrix_multiply)
result = matrix_multiply_jit(x, y)

print("결과 행렬 shape:", result.shape)
print("결과 행렬의 첫 few 요소:", result[:2, :2])