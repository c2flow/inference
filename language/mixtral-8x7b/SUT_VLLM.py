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
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8
    ):

        self.model_path = model_path or "mistralai/Mixtral-8x7B-Instruct-v0.1"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

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
        # Note: Mixtral may need different generation parameters
        # Using similar parameters as original SUT.py gen_kwargs
        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 1024,  # Matches max_new_tokens in original
            "min_tokens": 2,     # Matches min_new_tokens in original
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

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                tik1 = time.time()

                # Get dataset names for postProcess
                input_dataset = [self.data_object.dataset_names[q.index] for q in qitem]
                # Get input lengths for postProcess
                input_lens = [self.data_object.input_lens[q.index] for q in qitem]

                tik2 = time.time()

            # Original SUT.py uses tokenizer.batch_encode_plus to get input_ids
            # But vllm expects prompts (strings) or TokensPrompt objects
            # Since token IDs approaches fail in vllm 0.12.0, we use strings

            # Get original text inputs from dataset (like original SUT.py does)
            batch_texts = [self.data_object.input_texts[q.index] for q in qitem]

            # Use text prompts for vllm (most reliable approach)
            outputs = self.model.generate(
                prompts=batch_texts, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
                # log.info(f"Output: {output.outputs[0].text}")
            tik3 = time.time()

            # Convert to tensor for postProcess compatibility
            # Note: postProcess expects tensor with shape [batch_size, seq_len]
            max_len = max(len(tokens) for tokens in pred_output_tokens)
            padded_tokens = []
            for tokens in pred_output_tokens:
                if len(tokens) < max_len:
                    padded = tokens + [0] * (max_len - len(tokens))
                else:
                    padded = tokens
                padded_tokens.append(padded)

            out_tokens_tensor = torch.tensor(padded_tokens, dtype=torch.int64)

            # Call postProcess with required parameters for mixtral
            processed_output = self.data_object.postProcess(
                out_tokens_tensor,
                length=input_lens[0] if len(input_lens) == 1 else None,  # Handle batch case
                query_id_list=query_ids,
                dataset_list=input_dataset,
            )

            # Save to cache if not using cached outputs
            if not self.use_cached_outputs:
                os.makedirs("run_outputs", exist_ok=True)
                with _p.open(mode="wb") as f:
                    pickle.dump({"outputs": processed_output}, f)

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
                else:
                    log.info(f"\tLoaded from cache: {_p}")

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            distributed_executor_backend='mp',
            gpu_memory_utilization=0.9,
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
        total_sample_count=24576,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=8
    ):

        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.request_id = 0

        self.first_token_queue = queue.Queue()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

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

        outputs = output_response
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
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            # Mixtral dataset stores input_ids as tensors, need to convert
            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_ids_list = input_ids_tensor.cpu().numpy().flatten().tolist()

            input_dataset = [self.data_object.dataset_names[qitem.index]]
            input_len = self.data_object.input_lens[qitem.index]

            # TODO: This PoC is super slow with significant overhead. Best to
            # create a patch to `generate`
            results_generator = self.model.generate(
                prompt=TokensPrompt(prompt_token_ids=input_ids_list),
                sampling_params=self.sampling_params,
                request_id=str(self.request_id)
            )
            self.request_id += 1
            asyncio.run(self.stream_output(qitem, results_generator))

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()

    def load_model(self):
        log.info("Loading model")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size)
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")