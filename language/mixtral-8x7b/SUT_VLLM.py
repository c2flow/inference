import asyncio
import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Mixtral-8x7B-VLLM-SUT")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=24576,
        dataset_path=None,
        use_cached_outputs=False,
        workers=1,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        block_size=None,
        gpu_memory_utilization=0.9,
        enable_expert_parallel=False,
    ):

        self.model_path = model_path or "mistralai/Mixtral-8x7B-Instruct-v0.1"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_expert_parallel = enable_expert_parallel

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            device="cuda:0"  # vllm always uses GPU
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()
        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 1024,
            "min_tokens": 2,
        }
        self.sampling_params = SamplingParams(**gen_kwargs)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            # Check cache for each query individually
            cached_indices = []
            need_compute_indices = []
            cached_outputs = []

            if self.use_cached_outputs:
                for i, q in enumerate(qitem):
                    fname = f"run_outputs/q{q.index}.pkl"
                    if os.path.exists(fname):
                        with open(fname, "rb") as f:
                            cached = pickle.load(f)
                        cached_indices.append(i)
                        cached_outputs.append(cached["output"])
                    else:
                        need_compute_indices.append(i)
            else:
                need_compute_indices = list(range(len(qitem)))

            processed_output = [None] * len(qitem)

            for idx, output in zip(cached_indices, cached_outputs):
                processed_output[idx] = output

            tik1 = tik2 = tik3 = tok = None
            if need_compute_indices:
                tik1 = time.time()

                need_compute_qitems = [qitem[i] for i in need_compute_indices]

                input_dataset = [self.data_object.dataset_names[q.index] for q in need_compute_qitems]
                input_lens = [self.data_object.input_lens[q.index] for q in need_compute_qitems]

                tik2 = time.time()

                batch_texts = [self.data_object.input_texts[q.index] for q in need_compute_qitems]

                outputs = self.model.generate(
                    prompts=batch_texts, sampling_params=self.sampling_params
                )
                pred_output_tokens = []
                for output in outputs:
                    pred_output_tokens.append(list(output.outputs[0].token_ids))
                tik3 = time.time()

                max_len = max(len(tokens) for tokens in pred_output_tokens)
                padded_tokens = []
                for tokens in pred_output_tokens:
                    if len(tokens) < max_len:
                        padded = tokens + [0] * (max_len - len(tokens))
                    else:
                        padded = tokens
                    padded_tokens.append(padded)

                out_tokens_tensor = torch.tensor(padded_tokens, dtype=torch.int64)

                computed_output = self.data_object.postProcess(
                    out_tokens_tensor,
                    length=input_lens[0] if len(input_lens) == 1 else None,
                    query_id_list=[q.index for q in need_compute_qitems],
                    dataset_list=input_dataset,
                )

                for idx, output in zip(need_compute_indices, computed_output):
                    processed_output[idx] = output

                tok = time.time()

            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

                if cached_indices and need_compute_indices:
                    log.info(f"  (Loaded {len(cached_indices)} from cache, computed {len(need_compute_indices)})")
                elif cached_indices:
                    log.info(f"  (All {len(cached_indices)} loaded from cache)")

                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        if self.data_parallel_size != 1:
            raise NotImplementedError(
                "Data parallelism is not supported in Offline scenario. "
                "Use Server scenario for DP support."
            )
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            distributed_executor_backend=self.distributed_executor_backend,
            gpu_memory_utilization=self.gpu_memory_utilization,
            block_size=self.block_size,
            enable_expert_parallel=self.enable_expert_parallel,
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=24576,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        block_size=None,
        gpu_memory_utilization=0.9,
        enable_expert_parallel=False,
        **kwargs,
    ):

        super().__init__(
            model_path=model_path,
            dtype=dtype,
            batch_size=batch_size,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            block_size=block_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_expert_parallel=enable_expert_parallel,
        )
        self.request_id = 0
        self.request_id_lock = threading.Lock()

        self.first_token_queue = queue.Queue()

        # Shared asyncio event loop for AsyncLLMEngine
        self.event_loop = None
        self.event_loop_thread = None
        self.pending_futures = []
        self.pending_futures_lock = threading.Lock()

    def start(self):
        # Start shared event loop in a dedicated thread
        self.event_loop = asyncio.new_event_loop()
        self.event_loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True)
        self.event_loop_thread.start()

        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def _run_event_loop(self):
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    async def stream_output(self, qitem, results_generator):
        try:
            first = True
            async for request_output in results_generator:
                output_response = request_output
                if first:
                    first_tokens = list(output_response.outputs[0].token_ids)
                    response_data = array.array(
                        "B", np.array(first_tokens, np.int32).tobytes())
                    bi = response_data.buffer_info()
                    response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                    lg.FirstTokenComplete(response)
                    first = False

            pred_output_tokens = list(output_response.outputs[0].token_ids)
            n_tokens = len(pred_output_tokens)
            response_array = array.array(
                "B", np.array(pred_output_tokens, np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response = [
                lg.QuerySampleResponse(
                    qitem.id,
                    bi[0],
                    bi[1],
                    n_tokens)]
            lg.QuerySamplesComplete(response)
        except Exception:
            log.exception(f"Error processing query {qitem.index}")

    def process_queries(self):
        """Fire-and-forget submission to AsyncLLMEngine."""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            token_ids = self.data_object.input_ids[qitem.index]
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.squeeze(0).tolist()
            input_ids_tensor = TokensPrompt(prompt_token_ids=token_ids)

            with self.request_id_lock:
                request_id = self.request_id
                self.request_id += 1

            results_generator = self.model.generate(
                prompt=input_ids_tensor,
                sampling_params=self.sampling_params,
                request_id=str(request_id)
            )

            # Submit to event loop without blocking
            future = asyncio.run_coroutine_threadsafe(
                self.stream_output(qitem, results_generator),
                self.event_loop
            )
            with self.pending_futures_lock:
                self.pending_futures.append(future)

    def issue_queries(self, query_samples):
        for sample in query_samples:
            self.query_queue.put(sample)

    def stop(self):
        # Signal workers to stop
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join()

        # Wait for all pending async requests to complete
        with self.pending_futures_lock:
            futures = list(self.pending_futures)
        for future in futures:
            future.result()

        # Cancel all pending tasks and close event loop
        if self.event_loop is not None and self.event_loop_thread is not None:
            async def _cancel_all():
                tasks = [t for t in asyncio.all_tasks(self.event_loop)
                         if t is not asyncio.current_task(self.event_loop)]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            asyncio.run_coroutine_threadsafe(
                _cancel_all(), self.event_loop).result()

            self.event_loop.call_soon_threadsafe(self.event_loop.stop)
            self.event_loop_thread.join()
            self.event_loop.close()

    def load_model(self):
        log.info("Loading model")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            block_size=self.block_size,
            data_parallel_size=self.data_parallel_size,
            distributed_executor_backend=self.distributed_executor_backend,
            enable_expert_parallel=self.enable_expert_parallel,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")
