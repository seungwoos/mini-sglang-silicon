# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import time
from random import randint, seed

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark offline token generation.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model ID or local path.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-pages", type=int, help="KV cache page count (required on CPU/MPS).")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    is_cuda = args.device == "cuda"
    if min(args.num_seqs, args.max_input_len, args.max_output_len) <= 0:
        parser.error("Sequence count and token lengths must be positive.")
    if not is_cuda and (args.num_pages is None or args.num_pages <= 1):
        parser.error("CPU and MPS require --num-pages greater than 1.")

    page_size = 256 if is_cuda else 1
    max_seq_len = 4096
    if args.num_pages is not None:
        max_seq_len = min(max_seq_len, args.num_pages * page_size)
    if args.max_input_len + args.max_output_len > max_seq_len:
        parser.error(f"The maximum input and output lengths must fit in {max_seq_len} tokens.")

    seed(0)
    dtype = args.dtype or ("bfloat16" if is_cuda else "float32")
    llm = LLM(
        args.model,
        device=args.device,
        dtype=getattr(torch, dtype),
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        page_size=page_size,
        cache_type="radix" if is_cuda else "naive",
        num_page_override=args.num_pages,
    )

    try:
        max_token_id = min(10000, llm.engine.sampler.vocab_size - 1)
        min_input_len = min(100, args.max_input_len)
        min_output_len = min(100, args.max_output_len)
        prompt_token_ids = [
            [randint(0, max_token_id) for _ in range(randint(min_input_len, args.max_input_len))]
            for _ in range(args.num_seqs)
        ]
        sampling_params = [
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=True,
                max_tokens=randint(min_output_len, args.max_output_len),
            )
            for _ in range(args.num_seqs)
        ]
        llm.generate(
            [prompt_token_ids[0]],
            SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=2),
        )

        t = time.perf_counter()
        results = llm.generate(prompt_token_ids, sampling_params)
        elapsed = time.perf_counter() - t
        total_tokens = sum(len(result["token_ids"]) for result in results)
        throughput = total_tokens / elapsed
        print(f"Device: {args.device}, dtype: {dtype}, Requests: {len(results)}")
        print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f}tok/s")
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
