import fire
from typing import Union
from fastapi import FastAPI, Request
import uvicorn
from typing import Any, Dict, AnyStr, List, Union
from .flax_generator import FlaxHuggingfaceModel
from pydantic import BaseModel


class ChatRequest(BaseModel):
    greedy: bool = False
    response_prefix: str = None
    conversations: List[Dict[str, str]]

def main(
    model_name: str,
    host="0.0.0.0",
    port=35020,
    prompt_length=512,
    max_new_tokens=512,
    fully_sharded_data_parallel=True,
    mesh="fsdp"
    ):
    app = FastAPI()

    generator = FlaxHuggingfaceModel(
        model_name,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        fully_sharded_data_parallel=fully_sharded_data_parallel,
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

    @app.get("/")
    def read_root():
        return dict(
            model_name=model_name,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            fully_sharded_data_parallel=fully_sharded_data_parallel,
            mesh=mesh
        )


    @app.post("/chat")
    def chat(req: ChatRequest):
        print(req)
        return generator.chat(
            req.conversations,
            generation_prefix=req.response_prefix,
            greedy=req.greedy
            )
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    fire.Fire(main)