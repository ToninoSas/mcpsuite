"""
runner.py — Esecuzione inferenza su Qwen 4-bit quantizzato

Usa transformers + bitsandbytes (BitsAndBytes NF4/INT4) oppure
llama-cpp-python (GGUF) in base alla configurazione.

Il prompt viene formattato secondo il template Qwen2.5/Qwen3:
  - Il system prompt include i function schema in formato OpenAI
  - Il modello risponde con una <tool_call> oppure con testo

Per il residual stream capture (Fase 2), questo runner registrerà
anche forward hooks — per ora si limita a raccogliere l'output grezzo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from loader import BFCLSample


# ─────────────────────────────────────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerConfig:
    model_name_or_path: str  = "Qwen/Qwen3.5-9B"             # o path locale GGUF
    backend: str             = "transformers"                  # "transformers" | "llama_cpp"
    # BitsAndBytes
    load_in_4bit: bool       = True
    bnb_4bit_quant_type: str = "nf4"                           # "nf4" | "fp4"
    bnb_compute_dtype: str   = "bfloat16"
    # Generazione
    max_new_tokens: int      = 512
    temperature: float       = 0.0                             # greedy per riproducibilità
    do_sample: bool          = False
    # Troncamento input — fondamentale per sequenze multi-turn molto lunghe.
    # None = nessun limite; 3072 è sicuro su T4 16GB con NF4.
    max_seq_len: int | None  = None
    # Implementazione attenzione — "sdpa" usa PyTorch SDPA (più efficiente in
    # memoria per sequenze lunghe, senza dipendenze extra). "flash_attention_2"
    # richiede il pacchetto flash-attn. None lascia scegliere a transformers.
    attn_implementation: str | None = "sdpa"
    # Batch
    batch_size: int          = 1                               # 1 = sequenziale, sicuro su GPU 24GB
    # device_map: "auto" | {"": 0} | {"": 1} | ...
    # Per multi-GPU usa run_inference_parallel() che passa {"": gpu_id} automaticamente.
    device_map: str | dict   = "auto"


# ─────────────────────────────────────────────────────────────────────────────
# Formattazione del prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a helpful assistant with access to the following functions. \
Use them if required.

{functions_json}

Respond ONLY with a valid tool call in the format:
<tool_call>
{{"name": "<function_name>", "arguments": {{<json_args>}}}}
</tool_call>
If no function is appropriate, reply with: NO_CALL"""


def build_system_message(sample: BFCLSample) -> dict:
    """Restituisce il solo messaggio di sistema con gli schemi delle funzioni."""
    functions_json = json.dumps(sample.functions, indent=2, ensure_ascii=False)
    return {
        "role": "system",
        "content": SYSTEM_TEMPLATE.format(functions_json=functions_json),
    }


def build_prompt(sample: BFCLSample) -> list[dict]:
    """
    Costruisce la lista di messaggi (format chat) per un BFCLSample.

    Per single-turn: system + un singolo user message.
    Per multi-turn:  concatena tutti i turni (usato solo per inferenza single-pass).
                     Usa run_multi_turn_inference() per la valutazione corretta.
    """
    messages = [build_system_message(sample)]

    # sample.question: list[list[dict]] — ogni inner list è un turno
    for turn in sample.question:
        for msg in turn:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    return messages


def run_multi_turn_inference(
    sample: BFCLSample,
    runner: "TransformersRunner | LlamaCppRunner",
    capture_activations: bool = False,
) -> list[str]:
    """
    Esegue l'inferenza multi-turn chiamando il modello un turno alla volta,
    accumulando il contesto (incluse le risposte precedenti del modello).

    Se capture_activations=True e il runner supporta generate_with_hidden_state(),
    cattura l'hidden state dell'ultimo turno (contesto più ricco) e lo salva
    in sample.hidden_vec come numpy array shape (4096,) float16.

    Returns:
        Lista di output raw del modello, uno per turno.
    """
    messages: list[dict] = [build_system_message(sample)]
    outputs: list[str] = []
    n_turns = len(sample.question)

    for turn_idx, turn in enumerate(sample.question):
        for msg in turn:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        is_last = (turn_idx == n_turns - 1)
        if capture_activations and is_last and hasattr(runner, "generate_with_hidden_state"):
            import torch, numpy as np
            raw_output, hidden = runner.generate_with_hidden_state(messages)
            if hidden is not None:
                # Pool: last token del prefill dell'ultimo turno
                sample.hidden_vec = hidden[0, -1, :].to(torch.float16).numpy()
        else:
            raw_output = runner.generate(messages)

        outputs.append(raw_output)

        # Aggiunge la risposta del modello al contesto per il turno successivo
        messages.append({"role": "assistant", "content": raw_output})

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Runner Transformers (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

class TransformersRunner:
    """
    Carica Qwen con bitsandbytes 4-bit e genera le risposte.
    Predisposto per il forward-hook della Fase 2.
    """

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Carica modello e tokenizer (lazy, al primo uso)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        cfg = self.config
        print(f"[runner] Caricamento modello: {cfg.model_name_or_path}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_compute_dtype),
            bnb_4bit_use_double_quant=True,     # QLoRA double quant per risparmio memoria
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
        )

        load_kwargs: dict = dict(
            quantization_config=bnb_config,
            device_map=cfg.device_map,
            trust_remote_code=True,
        )
        if cfg.attn_implementation:
            load_kwargs["attn_implementation"] = cfg.attn_implementation

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path, **load_kwargs
            )
        except Exception:
            # sdpa o flash_attention_2 non supportati da questo modello: riprova senza
            load_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path, **load_kwargs
            )
        self.model.eval()
        self._loaded = True
        print("[runner] Modello caricato ✓")

    def generate(self, messages: list[dict]) -> str:
        """Genera la risposta dato un prompt in formato chat."""
        if not self._loaded:
            self.load()

        import torch

        # Applica il chat template di Qwen
        # enable_thinking=False: disabilita la thinking mode di Qwen3
        # (evita blocchi <think>...</think> che rallentano e non servono qui)
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Tronca a sinistra se la sequenza supera max_seq_len.
        # Mantiene i token più recenti (fine della conversazione).
        if self.config.max_seq_len and inputs["input_ids"].shape[1] > self.config.max_seq_len:
            inputs = {k: v[:, -self.config.max_seq_len:] for k, v in inputs.items()}

        cfg = self.config
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if cfg.do_sample else None,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Taglia il prompt dall'output
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Placeholder per Fase 2 ────────────────────────────────────────────────
    def generate_with_hidden_state(
        self,
        messages: list[dict],
    ) -> tuple[str, Any]:
        """
        Come generate(), ma cattura anche l'hidden state dell'ultimo layer.
        Usato nella Fase 2 per raccogliere le attivazioni.

        Returns:
            (raw_output_str, last_hidden_state_tensor)
        """
        if not self._loaded:
            self.load()

        import torch

        captured: dict[str, Any] = {}
        _prefill_done = [False]  # list to allow mutation inside nested function

        def _hook(module, input, output):
            # Capture only the prefill pass (first forward call).
            # model.generate() fires this hook once per token; subsequent calls
            # have seq_len=1 (decode step) and would overwrite the prefill state.
            if _prefill_done[0]:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            captured["last_hidden"] = hidden.detach().cpu()
            _prefill_done[0] = True

        # Registra l'hook sull'ultimo transformer block
        # Qwen2: model.model.layers[-1]
        last_layer = self.model.model.layers[-1]
        handle = last_layer.register_forward_hook(_hook)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        if self.config.max_seq_len and inputs["input_ids"].shape[1] > self.config.max_seq_len:
            inputs = {k: v[:, -self.config.max_seq_len:] for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=False,   # usiamo l'hook invece
            )

        handle.remove()

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return raw_output, captured.get("last_hidden", None)


# ─────────────────────────────────────────────────────────────────────────────
# Runner llama-cpp (GGUF)
# ─────────────────────────────────────────────────────────────────────────────

class LlamaCppRunner:
    """
    Alternativa con llama-cpp-python per modelli GGUF quantizzati.
    Nota: l'accesso agli hidden states è limitato in llama.cpp —
    preferire TransformersRunner per la Fase 2.
    """

    def __init__(self, config: RunnerConfig):
        self.config = config
        self.llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=self.config.model_name_or_path,
            n_ctx=8192,
            n_gpu_layers=-1,    # offload tutto su GPU
            verbose=False,
        )
        print("[runner] llama.cpp modello caricato ✓")

    def generate(self, messages: list[dict]) -> str:
        if self.llm is None:
            self.load()

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return output["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_runner(config: RunnerConfig) -> TransformersRunner | LlamaCppRunner:
    if config.backend == "transformers":
        return TransformersRunner(config)
    elif config.backend == "llama_cpp":
        return LlamaCppRunner(config)
    else:
        raise ValueError(f"Backend non supportato: {config.backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Funzione di alto livello
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_samples(
    samples: list[BFCLSample],
    runner: TransformersRunner | LlamaCppRunner,
    progress_every: int = 50,
    capture_activations: bool = False,
) -> list[BFCLSample]:
    """
    Esegue l'inferenza su tutti i sample, popolando `model_raw_output`.

    Per i sample single-turn: model_raw_output è una stringa.
    Per i sample multi-turn:  model_raw_output è una lista di stringhe
                               (una per turno), eseguita con contesto accumulato.

    Se capture_activations=True e il runner è TransformersRunner, popola anche
    sample.hidden_vec con il vettore (4096,) float16 catturato durante il prefill.
    Per i multi-turn viene catturato solo l'ultimo turno (contesto completo).

    Modifica i sample in-place e restituisce la lista.
    """
    import gc
    import torch
    from loader import MULTI_TURN_CATEGORIES

    if capture_activations and not hasattr(runner, "generate_with_hidden_state"):
        print("[runner] ⚠  capture_activations=True ma il runner non supporta "
              "generate_with_hidden_state() — le activations non verranno catturate.")
        capture_activations = False

    total = len(samples)
    errors = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        try:
            if sample.category in MULTI_TURN_CATEGORIES:
                sample.model_raw_output = run_multi_turn_inference(
                    sample, runner, capture_activations=capture_activations
                )
            else:
                messages = build_prompt(sample)
                if capture_activations:
                    import numpy as np
                    raw_output, hidden = runner.generate_with_hidden_state(messages)
                    sample.model_raw_output = raw_output
                    if hidden is not None:
                        # Pool: last token del prefill → vettore (4096,) float16
                        sample.hidden_vec = hidden[0, -1, :].to(torch.float16).numpy()
                else:
                    sample.model_raw_output = runner.generate(messages)
        except torch.cuda.OutOfMemoryError as exc:
            # Dopo OOM il contesto CUDA può essere parzialmente corrotto.
            # synchronize() attende che tutti i kernel pendenti finiscano
            # prima di liberare la memoria, prevenendo il cascading
            # "illegal memory access" sui sample successivi.
            print(f"[runner] ✗ OOM {sample.id} (seq troppo lunga?) — pulisco e continuo")
            sample.model_raw_output = "" if sample.category not in MULTI_TURN_CATEGORIES else []
            errors += 1
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[runner] ✗ sample {sample.id}: {exc}")
            sample.model_raw_output = "" if sample.category not in MULTI_TURN_CATEGORIES else []
            errors += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % progress_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"[runner] {i+1:>5}/{total}  "
                f"err={errors}  "
                f"{rate:.1f} sample/s  "
                f"ETA {eta/60:.1f}min"
            )

    return samples


def run_inference_parallel(
    samples: list[BFCLSample],
    config: RunnerConfig,
    num_gpus: int = 2,
    progress_every: int = 50,
    capture_activations: bool = False,
) -> list[BFCLSample]:
    """
    Esegue l'inferenza usando data parallelism su più GPU.

    Carica una copia del modello per ogni GPU, divide i sample equamente e
    processa ogni metà in un thread separato. Il threading funziona perché
    model.generate() rilascia il Python GIL durante le operazioni CUDA, quindi
    le due GPU lavorano davvero in parallelo (~2x throughput).

    Uso in pipeline o notebook:
        samples = run_inference_parallel(samples, runner_cfg, num_gpus=2)

    Args:
        samples:             lista di BFCLSample da processare
        config:              configurazione base (device_map viene ignorato)
        num_gpus:            numero di GPU da usare (default 2)
        progress_every:      frequenza del log di progresso per worker
        capture_activations: se True, cattura hidden_vec per ogni sample
    """
    import dataclasses
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    available = torch.cuda.device_count()
    if available < num_gpus:
        print(f"[runner] ⚠  richieste {num_gpus} GPU ma ne sono disponibili {available} — uso {available}")
        num_gpus = max(available, 1)

    if num_gpus == 1:
        runner = build_runner(config)
        return run_inference_on_samples(samples, runner, progress_every, capture_activations)

    # Divide i sample in chunk contigui (mantiene l'ordine)
    chunk_size = len(samples) // num_gpus
    chunks: list[list[BFCLSample]] = [
        samples[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_gpus - 1)
    ]
    chunks.append(samples[(num_gpus - 1) * chunk_size :])  # ultimo: eventuale resto

    # Caricamento sequenziale: from_pretrained + bitsandbytes non sono thread-safe.
    # Caricare i due modelli in sequenza evita race condition sull'inizializzazione
    # del contesto CUDA; l'inferenza vera e propria parte in parallelo dopo.
    runners: list[TransformersRunner] = []
    for gpu_id in range(num_gpus):
        gpu_config = dataclasses.replace(config, device_map={"": gpu_id})
        r = TransformersRunner(gpu_config)
        r.load()
        torch.cuda.synchronize(gpu_id)
        print(f"[runner] GPU {gpu_id} pronta — {len(chunks[gpu_id])} sample assegnati")
        runners.append(r)

    def _worker(runner: TransformersRunner, chunk: list[BFCLSample]) -> list[BFCLSample]:
        return run_inference_on_samples(chunk, runner, progress_every, capture_activations)

    results: dict[int, list[BFCLSample]] = {}
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(_worker, runners[gpu_id], chunks[gpu_id]): gpu_id
            for gpu_id in range(num_gpus)
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            results[gpu_id] = future.result()

    # Riassembla in ordine originale
    ordered: list[BFCLSample] = []
    for gpu_id in range(num_gpus):
        ordered.extend(results[gpu_id])
    return ordered
