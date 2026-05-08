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
from pathlib import Path
from typing import Any

from loader import BFCLSample


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_load(checkpoint_dir: Path) -> dict[str, dict]:
    """
    Legge tutti i sample già completati da un checkpoint.
    Cerca outputs.jsonl in checkpoint_dir e in tutte le sub-dir gpu*/
    (usate dalla modalità multi-GPU).
    Restituisce {sample_id: {"model_raw_output": ..., "has_act": bool}}.
    """
    done: dict[str, dict] = {}
    jsonl_paths = (
        list(checkpoint_dir.glob("outputs.jsonl")) +
        list(checkpoint_dir.glob("gpu*/outputs.jsonl"))
    )
    for path in jsonl_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass  # linea corrotta — ignora
    return done


def _ckpt_save(
    sample: BFCLSample,
    checkpoint_dir: Path,
    file_handle,
) -> None:
    """
    Salva atomicamente un sample appena completato.
    - hidden_vec → checkpoint_dir/acts/{id}.npy  (write-then-rename)
    - output     → outputs.jsonl  (append + flush immediato)
    """
    import numpy as np

    if getattr(sample, "hidden_vec", None) is not None:
        acts_dir = checkpoint_dir / "acts"
        acts_dir.mkdir(parents=True, exist_ok=True)
        safe_id    = sample.id.replace("/", "_").replace("\\", "_")
        tmp_path   = acts_dir / f"{safe_id}_tmp.npy"   # deve finire in .npy: np.save lo aggiunge altrimenti
        final_path = acts_dir / f"{safe_id}.npy"
        np.save(tmp_path, sample.hidden_vec)
        tmp_path.rename(final_path)

    file_handle.write(
        json.dumps({"id": sample.id, "model_raw_output": sample.model_raw_output},
                   ensure_ascii=False) + "\n"
    )
    file_handle.flush()


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
    # max_memory: forza la distribuzione del modello su più GPU.
    # Il cap deve essere ~metà del peso del modello, altrimenti device_map="auto"
    # mette tutto su GPU 0 (greedy fill).
    # Qwen3.5-9B NF4 ≈ 6.5 GB → cap consigliato: {0: "3500MiB", 1: "3500MiB"}
    # → ~16 layer per GPU, ~12.5 GB liberi per KV cache e attivazioni.
    max_memory: dict | None  = None


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
    # I messaggi system nelle live categories vengono saltati: usiamo il nostro
    for turn in sample.question:
        for msg in turn:
            if msg.get("role") == "system":
                continue
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
            raw_output, activations = runner.generate_with_hidden_state(messages)
            if activations is not None:
                # shape: (num_layers, hidden_size) float16
                sample.hidden_vec = activations
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
        if cfg.max_memory:
            load_kwargs["max_memory"] = cfg.max_memory
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

    def generate_with_hidden_state(
        self,
        messages: list[dict],
    ) -> tuple[str, "np.ndarray | None"]:
        """
        Come generate(), ma cattura l'ultimo token del prefill per ogni layer.

        Registra un hook su ciascuno dei 32 transformer block. Ogni hook
        viene eseguito appena il layer completa il forward pass durante il
        prefill: il vettore viene spostato immediatamente su CPU (float16)
        così la VRAM non accumula mai più di un hidden state alla volta.

        I decode step (seq_len == 1) vengono ignorati perché l'informazione
        rilevante per la predizione dell'allucinazione è nel prefill.

        Returns:
            (raw_output_str, activations)
            activations: numpy float16 shape (num_layers, hidden_size)
                         None se qualcosa è andato storto
        """
        if not self._loaded:
            self.load()

        import torch
        import numpy as np

        num_layers = len(self.model.model.layers)
        captured: dict[int, "torch.Tensor"] = {}

        def make_hook(layer_idx: int):
            def _hook(module, input, output):
                # seq_len > 1 → prefill pass; seq_len == 1 → decode step (ignora)
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[1] > 1 and layer_idx not in captured:
                    # Sposta subito su CPU float16: libera la VRAM immediatamente
                    captured[layer_idx] = hidden[0, -1, :].detach().to(torch.float16).cpu()
            return _hook

        handles = [
            layer.register_forward_hook(make_hook(i))
            for i, layer in enumerate(self.model.model.layers)
        ]

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
                output_hidden_states=False,
            )

        for h in handles:
            h.remove()

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if len(captured) == num_layers:
            activations = torch.stack(
                [captured[i] for i in range(num_layers)]
            ).numpy()                              # shape: (num_layers, hidden_size)
        else:
            activations = None

        return raw_output, activations


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
    checkpoint_dir: Path | None = None,
) -> list[BFCLSample]:
    """
    Esegue l'inferenza su tutti i sample, popolando `model_raw_output`.

    Per i sample single-turn: model_raw_output è una stringa.
    Per i sample multi-turn:  model_raw_output è una lista di stringhe
                               (una per turno), eseguita con contesto accumulato.

    Se capture_activations=True e il runner è TransformersRunner, popola anche
    sample.hidden_vec con il vettore (num_layers, hidden_size) float16.

    Se checkpoint_dir è impostato, ogni sample completato viene salvato
    immediatamente su disco. Al riavvio, i sample già presenti nel checkpoint
    vengono saltati e i loro output ripristinati automaticamente.

    Modifica i sample in-place e restituisce la lista.
    """
    import gc
    import numpy as np
    import torch
    from loader import MULTI_TURN_CATEGORIES

    if capture_activations and not hasattr(runner, "generate_with_hidden_state"):
        print("[runner] ⚠  capture_activations=True ma il runner non supporta "
              "generate_with_hidden_state() — le activations non verranno catturate.")
        capture_activations = False

    # ── Checkpoint: ripristina sample già completati ──────────────────────────
    done_ids: set[str] = set()
    ckpt_file = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        done_data = _ckpt_load(checkpoint_dir)
        if done_data:
            print(f"[runner] Checkpoint trovato: {len(done_data)} sample già completati — riprendo")
            for s in samples:
                if s.id in done_data:
                    s.model_raw_output = done_data[s.id]["model_raw_output"]
                    safe_id  = s.id.replace("/", "_").replace("\\", "_")
                    act_path = checkpoint_dir / "acts" / f"{safe_id}.npy"
                    if act_path.exists():
                        s.hidden_vec = np.load(act_path)
                    done_ids.add(s.id)

        ckpt_file = open(checkpoint_dir / "outputs.jsonl", "a", encoding="utf-8")

    total     = len(samples)
    n_skip    = len(done_ids)
    n_todo    = total - n_skip
    errors    = 0
    processed = 0
    t0        = time.time()

    if n_skip:
        print(f"[runner] {n_skip}/{total} sample saltati (già nel checkpoint), "
              f"rimangono {n_todo}")

    try:
        for i, sample in enumerate(samples):
            if sample.id in done_ids:
                continue

            try:
                if sample.category in MULTI_TURN_CATEGORIES:
                    sample.model_raw_output = run_multi_turn_inference(
                        sample, runner, capture_activations=capture_activations
                    )
                else:
                    messages = build_prompt(sample)
                    if capture_activations:
                        raw_output, activations = runner.generate_with_hidden_state(messages)
                        sample.model_raw_output = raw_output
                        if activations is not None:
                            sample.hidden_vec = activations
                    else:
                        sample.model_raw_output = runner.generate(messages)

            except torch.cuda.OutOfMemoryError:
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

            # Salva immediatamente (anche gli errori, per non riprocessarli)
            if ckpt_file is not None:
                _ckpt_save(sample, checkpoint_dir, ckpt_file)

            processed += 1

            if processed % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if processed % progress_every == 0 or processed == n_todo:
                elapsed = time.time() - t0
                rate    = processed / elapsed if elapsed > 0 else 0
                eta     = (n_todo - processed) / rate if rate > 0 else 0
                print(
                    f"[runner] {processed:>5}/{n_todo}  "
                    f"err={errors}  "
                    f"{rate:.1f} sample/s  "
                    f"ETA {eta/60:.1f}min"
                )
    finally:
        if ckpt_file is not None:
            ckpt_file.close()

    return samples


def run_inference_parallel(
    samples: list[BFCLSample],
    config: RunnerConfig,
    num_gpus: int = 2,
    progress_every: int = 50,
    capture_activations: bool = False,
    checkpoint_dir: Path | None = None,
) -> list[BFCLSample]:
    """
    Esegue l'inferenza usando data parallelism su più GPU.

    Carica una copia del modello per ogni GPU, divide i sample equamente e
    processa ogni metà in un thread separato. Il threading funziona perché
    model.generate() rilascia il Python GIL durante le operazioni CUDA, quindi
    le due GPU lavorano davvero in parallelo (~2x throughput).

    Per i sample multi-turn con sequenze molto lunghe è necessario impostare
    max_seq_len in RunnerConfig (es. 3072) per evitare OOM che corromperebbero
    il contesto CUDA rendendo la GPU inutilizzabile per i sample successivi.
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
        return run_inference_on_samples(
            samples, runner, progress_every, capture_activations, checkpoint_dir
        )

    # Divide i sample in chunk contigui (mantiene l'ordine)
    chunk_size = len(samples) // num_gpus
    chunks: list[list[BFCLSample]] = [
        samples[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_gpus - 1)
    ]
    chunks.append(samples[(num_gpus - 1) * chunk_size :])

    # Caricamento sequenziale: from_pretrained + bitsandbytes non sono thread-safe.
    runners: list[TransformersRunner] = []
    for gpu_id in range(num_gpus):
        gpu_config = dataclasses.replace(config, device_map={"": gpu_id})
        r = TransformersRunner(gpu_config)
        r.load()
        torch.cuda.synchronize(gpu_id)
        print(f"[runner] GPU {gpu_id} pronta — {len(chunks[gpu_id])} sample assegnati")
        runners.append(r)

    def _worker(
        runner: TransformersRunner,
        chunk: list[BFCLSample],
        gpu_id: int,
    ) -> list[BFCLSample]:
        # Ogni GPU scrive nel proprio sotto-dir per evitare conflitti su outputs.jsonl
        gpu_ckpt = (Path(checkpoint_dir) / f"gpu{gpu_id}") if checkpoint_dir else None
        return run_inference_on_samples(
            chunk, runner, progress_every, capture_activations, gpu_ckpt
        )

    results: dict[int, list[BFCLSample]] = {}
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(_worker, runners[gpu_id], chunks[gpu_id], gpu_id): gpu_id
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
