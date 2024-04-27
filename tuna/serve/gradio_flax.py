import fire
import gradio as gr
from typing import Any, Dict, AnyStr, List, Union
from .flax_generator import FlaxHuggingfaceModel


def main(
    model_name: str,
    host="0.0.0.0",
    port=35020,
    prompt_length=512,
    max_new_tokens=512,
    fully_sharded_data_parallel=True,
    chat_template: str = None,
    mesh="fsdp"
    ):

    generator = FlaxHuggingfaceModel(
        model_name,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        fully_sharded_data_parallel=fully_sharded_data_parallel,
        chat_template=chat_template,
        mesh_axes_shape=mesh,
    )
    print("Compiling...")

    print(generator.chat([{
        'role': 'user',
        'content': "Hi"
        }], greedy=True))
    print(generator.chat([{
        'role': 'user',
        'content': "Hi"
        }], greedy=False))



    def chat_function(message, history, system_prompt, greedy):
        # response = f"System prompt: {system_prompt}\n Message: {message}."
        convs = []
        if system_prompt:
            convs.append({
                'role': 'system',
                'content': system_prompt
            })
            
        for i, uttr in enumerate(history):
            convs.append({
                'role': 'user',
                'content': uttr[0]
            })
            convs.append({
                'role': 'assistant',
                'content': uttr[1]
            })
        convs.append({
            'role': 'user',
            'content': message
        })
        print(convs)
        response = generator.chat(convs, greedy=greedy)
        yield response

    demo = gr.ChatInterface(
        chat_function, 
        additional_inputs=[
            gr.Textbox("", label="System Prompt"), 
            # gr.Slider(10, 100)
            gr.Checkbox(False, label="Greedy")
        ],
        title=model_name,
        )
    demo.queue()
    demo.launch(
        debug=True,
        share=True,
        server_port=port,
        server_name=host
    )




if __name__ == "__main__":
    fire.Fire(main)