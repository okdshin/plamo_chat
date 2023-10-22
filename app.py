import argparse
import time
import socket
import json
import torch
from threading import Thread
from typing import Optional, Union, List

from flask import render_template
from flask import Flask
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class GenerateTextParams(BaseModel):
    input_text: str = ""
    max_new_tokens: int = 32
    do_sample: bool = True
    top_k: int = 100
    top_p: float = 0.9
    temperature: float = 0.5
    stream: bool = False


def make_streaming_response(json_gen):
    response = StreamingResponse(json_gen, media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    # response.headers["Connection"] = "keep-alive"
    return response


class ChatApp:
    def __init__(self, tokenizer_name: Optional[str], model_name: str, n_threads: int):
        self.model_name = model_name
        self.router = APIRouter()
        self.router.add_api_route("/api/v1/generate_text/", self.generate_text,
                                  methods=["POST"], response_model=None)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _generate_tokens(self, input_ids, params: GenerateTextParams) -> List[int]:
        return self.model.generate(
            inputs=torch.LongTensor([input_ids]),
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            use_cache=True,
        )[0]

    def _generate_text_stream(self, input_ids, params: GenerateTextParams) -> StreamingResponse:
        assert params.stream
        input_ids = self.tokenizer(params.input_text).input_ids

        def gen():
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(
                inputs=torch.LongTensor([input_ids]),
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                top_k=params.top_k,
                top_p=params.top_p,
                temperature=params.temperature,
                streamer=streamer,
                use_cache=True,
            )
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            for output in streamer:
                print(output)
                if not output:
                    continue
                yield json.dumps(dict(text=output))
            yield json.dumps(dict(text=None))

        return make_streaming_response(gen())

    def generate_text(self, params: GenerateTextParams) -> Union[JSONResponse, StreamingResponse]:
        print("received", params)
        input_ids = self.tokenizer(params.input_text).input_ids
        if params.stream:
            return self._generate_text_stream(input_ids, params)
        else:
            generated_tokens = self._generate_tokens(input_ids, params)
            generated_text = self.tokenizer.decode(generated_tokens)
            return JSONResponse(content=dict(text=generated_text))

    def docs(self):
        return RedirectResponse(url='/docs')


flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html", title="PLaMoChat")


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))  # Bind to a free port provided by the host.
        return int(s.getsockname()[1])  # Return the port number assigned.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)  # When None, assign automatically later
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--tokenizer-name", type=str)
    parser.add_argument("--open", type=str, choices=["none", "browser", "gui"], default="browser")
    parser.add_argument("--gui-scale", type=str, default="100%")
    args = parser.parse_args()
    print(args)

    if args.port is None:
        args.port = find_free_port()
    print(f"load from \"{args.model_name}\"")
    import uvicorn
    chat_app = ChatApp(n_threads=args.n_threads,
                       tokenizer_name=args.tokenizer_name, model_name=args.model_name)
    fast_api_app = FastAPI()
    fast_api_app.include_router(chat_app.router)
    fast_api_app.mount("/", WSGIMiddleware(flask_app))
    fast_api_app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    if args.open == "browser":
        import webbrowser
        from threading import Timer

        def open_browser():
            webbrowser.open(f"http://{args.host}:{args.port}", new=2)
        Timer(0.1, open_browser).start()
        uvicorn.run(fast_api_app, host=args.host, port=args.port, workers=1)
    elif args.open == "gui":
        from a2wsgi import ASGIMiddleware
        import webview
        window = webview.create_window('ChatApp', ASGIMiddleware(fast_api_app))

        def custom_logic(window):
            window.evaluate_js(f"document.documentElement.style.zoom = \"{args.gui_scale}\";")
        webview.start(custom_logic, window)
    else:
        assert args.open == "none"
        uvicorn.run(fast_api_app, host=args.host, port=args.port, workers=1)
