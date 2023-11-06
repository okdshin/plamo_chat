import argparse
import socket
import json
import torch
from threading import Thread
from typing import Any, Dict, Optional, Union, List

from flask import render_template
from flask import Flask
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
from transformers import (AutoTokenizer, TextIteratorStreamer, PreTrainedModel,
                          PretrainedConfig, PreTrainedTokenizer, StoppingCriteria)
from transformers.modeling_outputs import CausalLMOutput
import infer


class StopWord(StoppingCriteria):
    def __init__(self, stop_word: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.stop_tokens_len = len(tokenizer(stop_word).input_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        suffix_text = self.tokenizer.decode(input_ids[0][-self.stop_tokens_len:])
        return suffix_text.endswith(self.stop_word)


class PlamoCppConfig(PretrainedConfig):  # type: ignore
    model_type: str = "plamo_cpp"


class PlamoCppCausalLM(PreTrainedModel):

    def __init__(self, vocab_size, config: PlamoCppConfig):
        super().__init__(config)
        self.vocab_size = vocab_size

        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.plamo_cpp_model = infer.load_model_from_file(
            "/home/okada/plamo_cpp/plamo-13b/ggml-model-Q4_1.gguf", 8)  # infer.plamo_cpp_model()
        # "/home/okada/plamo_cpp/plamo-13b/ggml-model-f16.gguf", 8)  # infer.plamo_cpp_model()
        print(self.plamo_cpp_model)
        self.plamo_tokenzer = AutoTokenizer.from_pretrained(
            "pfnet/plamo-13b", trust_remote_code=True)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(  # type: ignore
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> CausalLMOutput:
        # logits = torch.rand(input_ids.size(0), 1, self.vocab_size)
        """
        out = self.model(input_ids)
        logits = out.logits
        print("dim", logits.dim())
        logits = logits[:, -1, :].unsqueeze(0)
        """
        logits = torch.from_numpy(self.plamo_cpp_model.calc_next_token_logits(
            input_ids.numpy(), self.vocab_size))
        # print("input_ids.size()", input_ids.size(), input_ids)
        print("input_ids decoded", self.plamo_tokenzer.decode(input_ids[0]))
        # print("logits.size()", logits.size(), logits)
        return CausalLMOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        model_inputs = {"input_ids": input_ids}
        return model_inputs


class GenerateTextParams(BaseModel):
    input_text: str = ""
    max_new_tokens: int = 128
    do_sample: bool = True
    top_k: int = 100
    top_p: float = 0.9
    temperature: float = 1.0
    exclude_input: bool = False
    stream: bool = False


def make_streaming_response(json_gen):
    response = StreamingResponse(json_gen, media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    # response.headers["Connection"] = "keep-alive"
    return response


class ChatApp:
    def __init__(
        self,
        tokenizer_name: Optional[str],
        model_name: str,
        prompt_template: str,
        stop_word: str,
        n_threads: int,
    ):
        self.model_name = model_name
        self.router = APIRouter()
        self.router.add_api_route("/api/v1/generate_text/", self.generate_text,
                                  methods=["POST"], response_model=None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = PlamoCppCausalLM(vocab_size=len(self.tokenizer), config=PlamoCppConfig())

        self.prompt_template = prompt_template
        print(f"stop_word `{stop_word}`")
        self.stop_word = StopWord(stop_word=stop_word, tokenizer=self.tokenizer)

    def _generate_tokens(self, input_ids, params: GenerateTextParams) -> List[int]:
        return self.model.generate(
            inputs=torch.LongTensor([input_ids]),
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            use_cache=True,
            stopping_criteria=[self.stop_word],
        )[0]

    def _generate_text_stream(self, input_ids, params: GenerateTextParams) -> StreamingResponse:
        assert params.stream

        def gen():
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=params.exclude_input)
            generation_kwargs = dict(
                inputs=torch.LongTensor([input_ids]),
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                top_k=params.top_k,
                top_p=params.top_p,
                temperature=params.temperature,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[self.stop_word],
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
        prompt = self.prompt_template.format(input_text=params.input_text)
        input_ids = self.tokenizer(prompt).input_ids
        if params.stream:
            return self._generate_text_stream(input_ids, params)
        else:
            generated_tokens = self._generate_tokens(input_ids, params)
            if params.exclude_input:
                generated_tokens = generated_tokens[len(input_ids):]
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
    parser.add_argument("--prompt-template-path", type=str)
    parser.add_argument("--stop-word", type=str)
    parser.add_argument("--open", type=str, choices=["none", "browser", "gui"], default="browser")
    parser.add_argument("--gui-scale", type=str, default="100%")
    args = parser.parse_args()
    print(args)

    args.stop_word = args.stop_word.replace("\\n", "\n")

    with open(args.prompt_template_path) as f:
        prompt_template = f.read()

    if args.port is None:
        args.port = find_free_port()
    print(f"load from \"{args.model_name}\"")
    import uvicorn
    chat_app = ChatApp(n_threads=args.n_threads,
                       tokenizer_name=args.tokenizer_name, model_name=args.model_name,
                       prompt_template=prompt_template, stop_word=args.stop_word)
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
