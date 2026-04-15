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
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        use_cached_outputs=False,
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        max_model_len=None,
        enable_chunked_prefill=False,
        block_size=None,
        max_seq_len_to_capture=None,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=None,
        max_num_seqs=1024
    ):

        self.model_path = model_path or f"meta-llama/Meta-Llama-3.1-8B-Instruct"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.max_model_len = max_model_len
        self.enable_chunked_prefill = enable_chunked_prefill
        self.block_size = block_size
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs

        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype
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
            "max_tokens": 128,
            "min_tokens": 1
        }
        self.sampling_params = SamplingParams(**gen_kwargs)
        # self.sampling_params.all_stop_token_ids.add(self.model.get_tokenizer().eos_token_id)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        # Create worker threads
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

            tik1 = time.time()

            input_ids_tensor = [
                TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index]) for q in qitem]
            # input_text_tensor = [
            #     self.data_object.input[q.index] for q in qitem]
            # for in_text in input_text_tensor:
            #     log.info(f"Input: {in_text}")

            tik2 = time.time()
            outputs = self.model.generate(
                prompts=input_ids_tensor, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
                # log.info(f"Output: {output.outputs[0].text}")
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
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

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
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
            max_model_len=self.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            block_size=self.block_size,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
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

        list_prompts_tokens = []
        list_prompts_attn_masks = []

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
        total_sample_count=13368,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        max_model_len=None,
        enable_chunked_prefill=False,
        block_size=None,
        max_seq_len_to_capture=None,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=None,
        max_num_seqs=1024,
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
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            block_size=block_size,
            max_seq_len_to_capture=max_seq_len_to_capture,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
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

    def process_queries(self):
        """Processor of the queued queries. Fire-and-forget submission to AsyncLLMEngine."""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = TokensPrompt(
                prompt_token_ids=self.data_object.input_ids[qitem.index])

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

        # Wait for workers to finish (queue will be drained before they see None)
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
            asyncio.run_coroutine_threadsafe(_cancel_all(), self.event_loop).result()

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
            max_model_len=self.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            block_size=self.block_size,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            data_parallel_size=self.data_parallel_size,
            distributed_executor_backend=self.distributed_executor_backend,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")
