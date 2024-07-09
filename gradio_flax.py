import fire
import gradio as gr
from typing import Any, Dict, AnyStr, List, Union
from eval.utils import load_model

def main(
    model_name: str,
    host="0.0.0.0",
    port=35020,
    prompt_length=1024,
    max_new_tokens=1024,
    eos_token: str = None,
    chat_template: str = None,
    ):

    generator = load_model(
        model_name,
        prompt_length=prompt_length,
        max_length=prompt_length + max_new_tokens,
        chat_template=chat_template,
        eos_token=eos_token,
        use_vllm=False
    )

    print("Compiling...")
    gen_args = {"max_new_tokens": 1024, "do_sample": False}
    print("greedy", generator.generate("Hi", gen_args=gen_args))
    
    gen_args["do_sample"] = True
    print("sample", generator.generate("Hi", gen_args=gen_args))
    
    
    use_stream = getattr(generator, "SUPPORT_STREAMING", False)

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
        print(convs)

        gen_args["do_sample"] = not greedy
        
        if use_stream:
            text = ""
            stop = False
            for token in generator.generate_stream(message, convs, gen_args=gen_args):
                for stop_token in ["<|endoftext|>", "<|im_end|>", "<end_of_turn>", "</s>", "<eos>"]:
                    if stop_token in token:
                        token = token.replace(stop_token, "")
                        stop = True
                text += token
                yield text

                if stop:
                    break
        else:
            response = generator.generate(message, convs, gen_args=gen_args)
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