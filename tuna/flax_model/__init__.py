import transformers as tf



from .mistral import FlaxMistralForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.MistralConfig,
    FlaxMistralForCausalLM,
    exist_ok=True
    )

from .phi3 import FlaxPhi3ForCausalLM
tf.FlaxAutoModelForCausalLM.register(
    tf.Phi3Config,
    FlaxPhi3ForCausalLM,
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