import atexit
from dataclasses import fields, replace
from time import perf_counter
from typing import Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True,
            local_files_only=True,
        )
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if sampling_params.stop and not sampling_params.stop_token_ids:
            sampling_params = replace(
                sampling_params,
                stop_token_ids=[
                    self.tokenizer.encode(stop, add_special_tokens=False)
                    for stop in sampling_params.stop
                    if stop
                ],
            )
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs = self.scheduler.schedule()
        token_ids, seq_need_compute_logits = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids, seq_need_compute_logits)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_total_tokens = sum(len(seq) for seq in seqs if seq.is_finished)
        num_generated_tokens = len(token_ids)
        return outputs, num_total_tokens, num_generated_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def score_loglikelihood(
        self,
        prompts: list[str] | list[list[int]],
        continuations: list[str] | list[list[int]],
    ) -> list[dict[str, float | int]]:
        token_id_seqs = []
        prompt_lens = []
        for prompt, continuation in zip(prompts, continuations):
            if isinstance(prompt, str):
                prompt_token_ids = self.tokenizer.encode(prompt)
            else:
                prompt_token_ids = list(prompt)
            if isinstance(continuation, str):
                continuation_token_ids = self.tokenizer.encode(
                    continuation,
                    add_special_tokens=False,
                )
            else:
                continuation_token_ids = list(continuation)
            token_id_seqs.append(prompt_token_ids + continuation_token_ids)
            prompt_lens.append(len(prompt_token_ids))
        return self.model_runner.call("score", token_id_seqs, prompt_lens)

    def score_rolling_loglikelihood(
        self,
        prompts: list[str] | list[list[int]],
    ) -> list[dict[str, float | int]]:
        max_window_tokens = max(self.scheduler.max_model_len, 2)
        step = max_window_tokens - 1
        token_id_seqs = []
        prompt_lens = []
        chunk_counts = []

        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_token_ids = self.tokenizer.encode(prompt)
            else:
                prompt_token_ids = list(prompt)

            if len(prompt_token_ids) <= 1:
                chunk_counts.append(0)
                continue

            count = 0
            start = 0
            while start < len(prompt_token_ids) - 1:
                end = min(len(prompt_token_ids), start + max_window_tokens)
                token_id_seqs.append(prompt_token_ids[start:end])
                prompt_lens.append(1)
                count += 1
                if end == len(prompt_token_ids):
                    break
                start += step
            chunk_counts.append(count)

        chunk_scores = self.model_runner.call("score", token_id_seqs, prompt_lens) if token_id_seqs else []
        results = []
        score_index = 0
        for count in chunk_counts:
            total_loglikelihood = 0.0
            total_tokens = 0
            for _ in range(count):
                chunk_result = chunk_scores[score_index]
                total_loglikelihood += float(chunk_result["loglikelihood"])
                total_tokens += int(chunk_result["num_tokens"])
                score_index += 1
            results.append(
                {
                    "loglikelihood": total_loglikelihood,
                    "num_tokens": total_tokens,
                    "avg_loglikelihood": (
                        total_loglikelihood / total_tokens if total_tokens > 0 else 0.0
                    ),
                }
            )
        return results

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        return_metrics: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, float | int]]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            seqs.append(self.add_request(prompt, sp))
        outputs = {}
        num_total_tokens = 0
        prompt_tokens = sum(seq.num_prompt_tokens for seq in seqs)
        t = perf_counter()
        while not self.is_finished():
            output, num_step_tokens, _ = self.step()
            num_total_tokens += num_step_tokens
            if use_tqdm:
                total_throughput = num_total_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "total_throughput": f"{int(total_throughput)}tok/s",
                })
            
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        if not return_metrics:
            return outputs
        completion_tokens = sum(len(output["token_ids"]) for output in outputs)
        total_time = perf_counter() - t
        metrics = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_time": total_time,
        }
        return outputs, metrics
