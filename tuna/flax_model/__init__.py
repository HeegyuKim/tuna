import transformers as tf



from .mistral import FlaxMistralForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.MistralConfig,
    FlaxMistralForCausalLM,
    exist_ok=True
    )
    
from .phi import FlaxPhiForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.PhiConfig,
    FlaxPhiForCausalLM,
    exist_ok=True
    )

from .phi3 import FlaxPhi3ForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.Phi3Config,
    FlaxPhi3ForCausalLM,
    exist_ok=True
    )

from .qwen2 import FlaxQwen2ForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.Qwen2Config,
    FlaxQwen2ForCausalLM,
    exist_ok=True
    )


# from .gemma import FlaxGemmaForCausalLM
# tf.FlaxAutoModelForCausalLM.register(
#     tf.GemmaConfig,
#     FlaxGemmaForCausalLM,
#     exist_ok=True
#     )
    
# from .llama import FlaxLlamaForCausalLM
# tf.FlaxAutoModelForCausalLM.register(
#     tf.LlamaConfig,
#     FlaxLlamaForCausalLM,
#     exist_ok=True
#     )