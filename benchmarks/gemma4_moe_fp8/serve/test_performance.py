#!/usr/bin/env python3
"""Transparent online serving benchmark for Gemma4 E011.

Sends requests to vLLM OpenAI-compatible endpoint via HTTP,
measures throughput and latency with full visibility into the logic.

Usage:
    python test_performance.py --host localhost --port 8100 --num-prompts 1000
    python test_performance.py --dataset datasets/sc1_delta_v2.jsonl --request-rate 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp


@dataclass
class RequestResult:
    prompt_len: int = 0
    output_len: int = 0
    ttft: float = 0.0          # time to first token (s)
    total_time: float = 0.0    # total request time (s)
    success: bool = False
    error: str = ""


@dataclass
class BenchmarkResult:
    num_requests: int = 0
    num_success: int = 0
    total_time: float = 0.0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    results: list[RequestResult] = field(default_factory=list)

    def report(self):
        if not self.num_success:
            print("ERROR: No successful requests!")
            return

        ttfts = sorted(r.ttft for r in self.results if r.success)
        tpots = sorted(
            (r.total_time - r.ttft) / max(r.output_len - 1, 1)
            for r in self.results if r.success and r.output_len > 1
        )
        req_times = sorted(r.total_time for r in self.results if r.success)

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        throughput_out = self.total_output_tokens / self.total_time
        throughput_total = (self.total_prompt_tokens + self.total_output_tokens) / self.total_time

        print(f"\n{'='*60}")
        print(f"  RESULTS ({self.num_success}/{self.num_requests} successful)")
        print(f"{'='*60}")
        print(f"  Total time:          {self.total_time:.2f} s")
        print(f"  Requests/s:          {self.num_success / self.total_time:.2f}")
        print(f"  Output tok/s:        {throughput_out:.1f}")
        print(f"  Total tok/s:         {throughput_total:.1f}")
        print(f"  Prompt tokens:       {self.total_prompt_tokens}")
        print(f"  Output tokens:       {self.total_output_tokens}")
        print()
        print(f"  TTFT (s):   p50={percentile(ttfts,50):.3f}  p90={percentile(ttfts,90):.3f}  p99={percentile(ttfts,99):.3f}")
        if tpots:
            print(f"  TPOT (s):   p50={percentile(tpots,50):.4f}  p90={percentile(tpots,90):.4f}  p99={percentile(tpots,99):.4f}")
        print(f"  Latency(s): p50={percentile(req_times,50):.2f}  p90={percentile(req_times,90):.2f}  p99={percentile(req_times,99):.2f}")
        print(f"{'='*60}\n")


def load_prompts(dataset_path: str, num_prompts: int) -> list[str]:
    """Load prompts from JSONL file (same format as bench_ablation.py)."""
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Support both {"prompt": "..."} and {"messages": [...]} formats
            if "prompt" in d:
                prompts.append(d["prompt"])
            elif "messages" in d:
                # Fold messages into single prompt
                parts = []
                for m in d["messages"]:
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    parts.append(content)
                prompts.append("\n".join(parts))
            if len(prompts) >= num_prompts:
                break
    print(f"Loaded {len(prompts)} prompts from {dataset_path}")
    return prompts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool = True,
) -> RequestResult:
    """Send one chat completion request and measure timing."""
    result = RequestResult()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    t_start = time.perf_counter()
    first_token_received = False

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                result.total_time = time.perf_counter() - t_start
                return result

            if stream:
                output_tokens = 0
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and not first_token_received:
                        result.ttft = time.perf_counter() - t_start
                        first_token_received = True
                    if content:
                        output_tokens += 1  # approximate: 1 SSE chunk ≈ 1 token

                result.output_len = output_tokens
            else:
                data = await resp.json()
                result.ttft = time.perf_counter() - t_start
                usage = data.get("usage", {})
                result.output_len = usage.get("completion_tokens", 0)
                result.prompt_len = usage.get("prompt_tokens", 0)

            result.total_time = time.perf_counter() - t_start
            result.success = True

    except Exception as e:
        result.error = str(e)
        result.total_time = time.perf_counter() - t_start

    return result


async def benchmark(
    prompts: list[str],
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    request_rate: float,
    max_concurrency: int,
    stream: bool,
) -> BenchmarkResult:
    """Run benchmark: send all prompts at specified rate."""
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def throttled_request(session, prompt, delay):
        if delay > 0:
            await asyncio.sleep(delay)
        if semaphore:
            async with semaphore:
                return await send_request(session, url, prompt, model, max_tokens, temperature, stream)
        return await send_request(session, url, prompt, model, max_tokens, temperature, stream)

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            if request_rate == float("inf"):
                delay = 0
            else:
                delay = i / request_rate
            tasks.append(throttled_request(session, prompt, delay))

        print(f"Sending {len(prompts)} requests (rate={'inf' if request_rate == float('inf') else request_rate} req/s)...")
        t_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    bench_result = BenchmarkResult(
        num_requests=len(results),
        num_success=sum(1 for r in results if r.success),
        total_time=total_time,
        total_prompt_tokens=sum(r.prompt_len for r in results),
        total_output_tokens=sum(r.output_len for r in results),
        results=results,
    )
    return bench_result


def main():
    parser = argparse.ArgumentParser(description="Gemma4 E011 Online Benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to JSONL dataset (default: datasets/sc1_delta_v2.jsonl)")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Requests per second (default: inf = all at once)")
    parser.add_argument("--max-concurrency", type=int, default=0,
                        help="Max concurrent requests (0 = unlimited)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Use non-streaming mode (get usage stats from response)")
    args = parser.parse_args()

    # Resolve dataset path
    if args.dataset is None:
        script_dir = Path(__file__).parent
        args.dataset = str(script_dir / ".." / "datasets" / "sc1_delta_v2.jsonl")

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    prompts = load_prompts(args.dataset, args.num_prompts)

    print(f"\n{'='*60}")
    print(f"  Gemma4 E011 Online Serving Benchmark")
    print(f"{'='*60}")
    print(f"  Server:       {url}")
    print(f"  Model:        {args.model}")
    print(f"  Prompts:      {len(prompts)}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Request rate: {'inf' if args.request_rate == float('inf') else args.request_rate} req/s")
    print(f"  Concurrency:  {'unlimited' if args.max_concurrency == 0 else args.max_concurrency}")
    print(f"  Stream:       {not args.no_stream}")
    print(f"{'='*60}\n")

    result = asyncio.run(benchmark(
        prompts=prompts,
        url=url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        stream=not args.no_stream,
    ))
    result.report()

    # Print errors if any
    errors = [r for r in result.results if not r.success]
    if errors:
        print(f"\n  {len(errors)} failed requests:")
        for e in errors[:5]:
            print(f"    - {e.error[:100]}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")


if __name__ == "__main__":
    main()
