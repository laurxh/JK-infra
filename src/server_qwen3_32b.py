import argparse
import json
import os
import queue
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "qwen",
    "Qwen3-32B",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve Qwen3-32B with serialized generation."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the HTTP server.",
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="Local Qwen3-32B model directory.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length used by the engine.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help="Maximum concurrent sequences configured in the engine.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=32768,
        help="Maximum batched tokens configured in the engine.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.97,
        help="Fraction of each GPU memory reserved for weights and KV cache.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture. Disabled by default.",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        default=600.0,
        help="Maximum time in seconds for a request to wait for completion.",
    )
    return parser.parse_args()


@dataclass
class GenerateTask:
    request_id: str | int | None
    prompt: str | list[int]
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    ignore_eos: bool
    stop: list[str]
    created_at: float
    done: threading.Event
    prompt_tokens: int | None = None
    submitted_to_engine_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None


class GenerationWorker:
    def __init__(self, args):
        from nanovllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self._llm_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        model_path = os.path.abspath(os.path.expanduser(args.model_path))
        self.llm = LLM(
            model_path,
            tensor_parallel_size=4,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=not args.cuda_graph,
        )
        self.tasks: queue.Queue[GenerateTask | None] = queue.Queue()
        self.queue_timeout = args.queue_timeout
        self.shutdown_event = threading.Event()
        self._queue_stats_snapshot = self._empty_queue_stats()

        self.thread = threading.Thread(target=self._loop, name="qwen3-worker", daemon=True)
        self.thread.start()

    def submit(self, task: GenerateTask):
        self.tasks.put(task)

    def score_loglikelihood(
        self,
        prompt: str | list[int],
        continuation: str | list[int],
    ) -> dict[str, float | int]:
        with self._llm_lock:
            result = self.llm.score_loglikelihood([prompt], [continuation])[0]
        return result

    def score_rolling_loglikelihood(
        self,
        prompt: str | list[int],
    ) -> dict[str, float | int]:
        with self._llm_lock:
            result = self.llm.score_rolling_loglikelihood([prompt])[0]
        return result

    def close(self):
        self.shutdown_event.set()
        self.tasks.put(None)
        self.thread.join(timeout=5)
        try:
            with self._llm_lock:
                self.llm.exit()
        except Exception:
            pass

    @staticmethod
    def _empty_queue_stats() -> dict:
        return {
            "running": {
                "task_count": 0,
                "decode_tokens_remaining": 0,
            },
            "waiting": {
                "task_count": 0,
                "compute_tokens_remaining": 0,
            },
        }

    def _collect_engine_queue_stats_locked(self) -> dict:
        scheduler = self.llm.scheduler
        running = list(scheduler.running)
        waiting = list(scheduler.waiting)
        return {
            "running": {
                "task_count": len(running),
                "decode_tokens_remaining": sum(
                    max(seq.max_tokens - seq.num_completion_tokens, 0)
                    for seq in running
                ),
            },
            "waiting": {
                "task_count": len(waiting),
                "compute_tokens_remaining": sum(
                    max(len(seq) - seq.num_cached_tokens, 0)
                    + max(seq.max_tokens - seq.num_completion_tokens, 0)
                    for seq in waiting
                ),
            },
        }

    def _set_queue_stats_snapshot(self, stats: dict):
        with self._stats_lock:
            self._queue_stats_snapshot = stats

    @staticmethod
    def _truncate_text_at_stop(text: str, stop: list[str]) -> str:
        if not stop:
            return text
        end = len(text)
        for item in stop:
            index = text.find(item)
            if index != -1:
                end = min(end, index)
        return text[:end]

    @classmethod
    def _build_result(cls, task: GenerateTask, text: str) -> dict:
        response = {
            "text": cls._truncate_text_at_stop(text, task.stop),
        }
        if task.request_id is not None:
            response["ID"] = task.request_id
        return response

    def get_queue_stats(self) -> dict:
        if hasattr(self, "_stats_lock") and hasattr(self, "_queue_stats_snapshot"):
            with self._stats_lock:
                return {
                    "running": dict(self._queue_stats_snapshot["running"]),
                    "waiting": dict(self._queue_stats_snapshot["waiting"]),
                }
        with self._llm_lock:
            return self._collect_engine_queue_stats_locked()

    def _loop(self):
        active_tasks: dict[int, tuple[GenerateTask, object]] = {}
        while not self.shutdown_event.is_set():
            pending_tasks: list[GenerateTask] = []

            if not active_tasks:
                try:
                    task = self.tasks.get(timeout=0.1)
                except queue.Empty:
                    self._set_queue_stats_snapshot(self._empty_queue_stats())
                    continue
                if task is None:
                    break
                pending_tasks.append(task)

            while True:
                try:
                    task = self.tasks.get_nowait()
                except queue.Empty:
                    break
                if task is None:
                    self.shutdown_event.set()
                    break
                pending_tasks.append(task)

            failed_tasks: list[tuple[GenerateTask, str]] = []
            completed_outputs: list[tuple[int, list[int], str]] = []

            try:
                if pending_tasks:
                    with self._llm_lock:
                        for task in pending_tasks:
                            try:
                                sampling_params = self._SamplingParams(
                                    temperature=task.temperature,
                                    max_tokens=task.max_tokens,
                                    top_p=task.top_p,
                                    top_k=task.top_k,
                                    ignore_eos=task.ignore_eos,
                                    stop=task.stop,
                                )
                                seq = self.llm.add_request(task.prompt, sampling_params)
                                task.prompt_tokens = seq.num_prompt_tokens
                                task.submitted_to_engine_at = time.time()
                                active_tasks[seq.seq_id] = (task, seq)
                            except Exception:
                                failed_tasks.append((task, traceback.format_exc()))
                        self._set_queue_stats_snapshot(self._collect_engine_queue_stats_locked())

                if active_tasks:
                    with self._llm_lock:
                        outputs, _, _ = self.llm.step()
                        step_finished_at = time.time()
                        for task, seq in active_tasks.values():
                            if task.first_token_at is None and seq.num_completion_tokens > 0:
                                task.first_token_at = step_finished_at
                        completed_outputs = [
                            (seq_id, self.llm.tokenizer.decode(token_ids))
                            for seq_id, token_ids in outputs
                        ]
                        self._set_queue_stats_snapshot(self._collect_engine_queue_stats_locked())
            except Exception:
                error = traceback.format_exc()
                failed_tasks.extend((task, error) for task, _ in active_tasks.values())
                active_tasks.clear()
                self._set_queue_stats_snapshot(self._empty_queue_stats())

            for task, error in failed_tasks:
                task.error = error
                task.done.set()

            for seq_id, text in completed_outputs:
                active_task = active_tasks.pop(seq_id, None)
                if active_task is None:
                    continue
                task, _ = active_task
                if task.first_token_at is None and text:
                    task.first_token_at = time.time()
                task.finished_at = time.time()
                task.result = self._build_result(task, text)
                task.done.set()


class ServerState:
    def __init__(self, worker: GenerationWorker):
        self.worker = worker
        self.started_at = time.time()


def make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Qwen3HTTP/1.0"

        @staticmethod
        def _parse_text_or_token_ids(
            payload: dict,
            text_key: str,
            token_ids_key: str,
            empty_error: str,
        ) -> str | list[int] | tuple[None, str]:
            text_value = payload.get(text_key)
            token_ids_value = payload.get(token_ids_key)
            if text_value is not None and token_ids_value is not None:
                return None, f"provide either {text_key} or {token_ids_key}, not both"
            if token_ids_value is not None:
                if (
                    not isinstance(token_ids_value, list)
                    or not token_ids_value
                    or any(
                        not isinstance(token_id, int) or token_id < 0
                        for token_id in token_ids_value
                    )
                ):
                    return None, f"{token_ids_key} must be a non-empty list of non-negative integers"
                return token_ids_value
            if isinstance(text_value, str) and text_value:
                return text_value
            return None, empty_error

        def do_GET(self):
            if self.path == "/health":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "uptime": time.time() - state.started_at,
                    },
                )
                return
            if self.path == "/stats":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "queue_stats": state.worker.get_queue_stats(),
                    },
                )
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self):
            if self.path == "/loglikelihood_rolling":
                self._handle_loglikelihood_rolling()
                return
            if self.path == "/loglikelihood":
                self._handle_loglikelihood()
                return
            if self.path != "/generate":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                return
            if content_length <= 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "empty_body"})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return

            request_id = payload.get("ID", payload.get("id"))
            request_prompt = payload.get("prompt")
            if not isinstance(request_prompt, str) or not request_prompt:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "prompt must be a non-empty string"},
                )
                return

            temperature = payload.get("temperature", 0.7)
            max_tokens = payload.get("max_tokens", payload.get("max_gen_toks", 256))
            top_p = payload.get("top_p", 1.0)
            top_k = payload.get("top_k", 0)
            ignore_eos = payload.get("ignore_eos", False)
            stop = payload.get("stop", payload.get("until", []))
            if not isinstance(temperature, (int, float)) or temperature < 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "temperature must be >= 0"})
                return
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "max_tokens must be a positive integer"})
                return
            if not isinstance(top_p, (int, float)) or not (0 < float(top_p) <= 1.0):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "top_p must be in (0, 1]"})
                return
            if not isinstance(top_k, int) or top_k < 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "top_k must be a non-negative integer"})
                return
            if not isinstance(ignore_eos, bool):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "ignore_eos must be a boolean"})
                return
            if isinstance(stop, str):
                stop = [stop]
            if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "stop/until must be a string or a list of strings"},
                )
                return

            task = GenerateTask(
                request_id=request_id,
                prompt=request_prompt,
                temperature=float(temperature),
                max_tokens=max_tokens,
                top_p=float(top_p),
                top_k=top_k,
                ignore_eos=ignore_eos,
                stop=stop,
                created_at=time.time(),
                done=threading.Event(),
            )
            state.worker.submit(task)

            if not task.done.wait(timeout=state.worker.queue_timeout):
                self._write_json(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    {
                        "error": "request_timeout",
                        "message": "The request did not finish before queue_timeout.",
                    },
                )
                return

            if task.error is not None:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "generation_failed",
                        "message": task.error,
                    },
                )
                return

            self._write_json(HTTPStatus.OK, task.result)

        def _handle_loglikelihood(self):
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                return
            if content_length <= 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "empty_body"})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return

            eval_request_type = payload.get("eval_request_type")
            if eval_request_type is not None and eval_request_type != "loglikelihood":
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "eval_request_type must be loglikelihood for this endpoint"},
                )
                return

            request_id = payload.get("ID", payload.get("id"))

            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "prompt must be a non-empty string"})
                return

            continuation = payload.get("eval_continuation")
            if not isinstance(continuation, str) or not continuation:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "eval_continuation must be a non-empty string"},
                )
                return

            try:
                result = state.worker.score_loglikelihood(prompt, continuation)
            except Exception:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "loglikelihood_failed",
                        "message": traceback.format_exc(),
                    },
                )
                return

            response = {"accuracy": result["loglikelihood"]}
            if request_id is not None:
                response["ID"] = request_id
            self._write_json(HTTPStatus.OK, response)

        def _handle_loglikelihood_rolling(self):
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                return
            if content_length <= 0:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "empty_body"})
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return

            eval_request_type = payload.get("eval_request_type")
            if eval_request_type is not None and eval_request_type != "loglikelihood_rolling":
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "eval_request_type must be loglikelihood_rolling for this endpoint"},
                )
                return

            request_id = payload.get("ID", payload.get("id"))
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": "prompt must be a non-empty string"})
                return

            try:
                result = state.worker.score_rolling_loglikelihood(prompt)
            except Exception:
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": "loglikelihood_rolling_failed",
                        "message": traceback.format_exc(),
                    },
                )
                return

            response = {"accuracy": result["loglikelihood"]}
            if request_id is not None:
                response["ID"] = request_id
            self._write_json(HTTPStatus.OK, response)

        def log_message(self, format, *args):
            return

        def _write_json(self, status: HTTPStatus, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    startup_begin = time.time()
    args = parse_args()
    worker = GenerationWorker(args)
    state = ServerState(worker)
    handler_cls = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    startup_total = time.time() - startup_begin

    def shutdown_handler(signum, frame):
        del signum, frame
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"Serving Qwen3-32B on http://{args.host}:{args.port}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(
        "Engine config: "
        f"max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}"
    )
    print(f"Startup time: {startup_total:.2f}s")
    print(
        "Endpoints: GET /health, GET /stats, "
        "POST /generate, POST /loglikelihood, POST /loglikelihood_rolling"
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()
