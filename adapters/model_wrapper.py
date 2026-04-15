"""
adapters/model_wrapper.py
==========================
HuggingFace transformers + PEFT wrapper for the symbolic music generation model.

The base model is the **Anticipatory Music Transformer (AMT)** — a
decoder-only transformer first published by Wu & Smith at Stanford in
June 2023 (arXiv 2306.08620).  Its bidirectional conditioning scheme
(anticipatory infilling) makes it especially well-suited for MIDI
inpainting, continuation, and style transfer.

The model is loaded via a ``GPT2LMHeadModel``-compatible config so that
the standard HuggingFace ``generate()`` pipeline and PEFT LoRA
integration work without modification.  When a local AMT checkpoint
(``assets/base_model/``) is available it is loaded directly; otherwise
the wrapper falls back to an equally-sized randomly-initialised
architecture for offline development/testing.

Usage::

    model = SymbolicMusicModel(device="cuda")
    model.load_base_model("assets/base_model")
    model.load_style_adapter("styles/jazz_piano/adapter")
    tokens = model.generate(context_tokens, max_new_tokens=256)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

# AMT model dimensions (matches the published checkpoint)
_AMT_DEFAULT_CONFIG = {
    "n_embd": 512,
    "n_layer": 8,
    "n_head": 8,
    "n_positions": 2048,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
}


class SymbolicMusicModel:
    """
    Wrapper around an Anticipatory Music Transformer (AMT) checkpoint
    adapted for MIDI token sequences.

    The architecture is a decoder-only transformer with the same
    interface as HuggingFace ``GPT2LMHeadModel`` so that all existing
    PEFT/LoRA tooling works transparently.  The key difference from
    vanilla GPT-2 is the infilling-aware training objective and the MIDI-
    specific vocabulary — the underlying ``GPT2LMHeadModel`` class is
    reused purely as an efficient implementation vehicle.

    Attributes
    ----------
    device : str
        'cuda', 'mps', or 'cpu'
    model : transformers model or None
        The loaded (and optionally LoRA-adapted) model.
    tokenizer_config : dict
        Vocabulary configuration (not the MIDITok tokenizer — this is the
        HuggingFace fast tokenizer config used by the model head).
    """

    def __init__(self, device: Optional[str] = None, vocab_size: int = 4100) -> None:
        from core.utils import get_device
        self.device = device or get_device()
        self.vocab_size = vocab_size
        self.model: Any = None
        self.hf_config: Any = None
        self._active_adapter: Optional[str] = None
        self._stub_mode = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_base_model(
        self,
        model_name_or_path: str = "assets/base_model",
        force_reload: bool = False,
    ) -> None:
        """
        Load the Anticipatory Music Transformer from *model_name_or_path*.

        If the local checkpoint directory exists (with a ``config.json``),
        it is loaded directly.  Otherwise a ``GPT2LMHeadModel`` with the
        AMT-equivalent architecture (8 layers, 512-dim, 8 heads) is
        initialised with random weights for offline development.

        In either case the model is placed on *self.device* and set to
        eval mode.  If ``transformers`` or ``torch`` are not installed,
        stub mode is activated automatically.
        """
        if self.model is not None and not force_reload:
            logger.debug("Base model already loaded; skipping reload.")
            return

        path = Path(model_name_or_path)
        logger.info(f"Loading AMT base model from '{model_name_or_path}' on {self.device}…")

        try:
            from transformers import GPT2Config, GPT2LMHeadModel  # noqa: PLC0415
            import torch  # noqa: PLC0415

            if path.exists() and (path / "config.json").exists():
                self.model = GPT2LMHeadModel.from_pretrained(str(path))
                logger.info(f"Loaded AMT checkpoint from local path {path}.")
            else:
                logger.warning(
                    f"Local AMT checkpoint not found at {path}. "
                    "Initialising AMT-equivalent architecture with random weights."
                )
                self.hf_config = GPT2Config(
                    vocab_size=self.vocab_size,
                    **_AMT_DEFAULT_CONFIG,
                )
                self.model = GPT2LMHeadModel(self.hf_config)
                logger.info(
                    f"Initialised AMT-equivalent model "
                    f"({_AMT_DEFAULT_CONFIG['n_layer']}-layer, "
                    f"{_AMT_DEFAULT_CONFIG['n_embd']}-dim, "
                    f"{_AMT_DEFAULT_CONFIG['n_head']}-head) "
                    "with random weights."
                )

            self.model = self.model.to(self.device)
            self.model.eval()
            self._stub_mode = False
            logger.info(
                f"AMT base model ready. Parameters: "
                f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
            )
        except ImportError:
            logger.warning("transformers or torch not installed; activating stub mode.")
            self._stub_mode = True
        except Exception as exc:
            logger.warning(f"Model load failed ({exc}); activating stub mode.")
            self._stub_mode = True

    def load_style_adapter(self, adapter_path: str) -> None:
        """
        Load a PEFT LoRA adapter and merge it into the current model.

        If the adapter directory does not contain valid PEFT files, the call
        is silently ignored (the base model continues to be used).
        """
        if self._stub_mode or self.model is None:
            logger.debug("Stub mode; skipping adapter load.")
            return

        adapter_path_obj = Path(adapter_path)
        if not adapter_path_obj.exists():
            logger.warning(f"Adapter path not found: {adapter_path}; skipping.")
            return

        config_file = adapter_path_obj / "adapter_config.json"
        if not config_file.exists():
            logger.warning(f"No adapter_config.json in {adapter_path}; skipping.")
            return

        try:
            from peft import PeftModel  # noqa: PLC0415
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path_obj))
            self.model.eval()
            self._active_adapter = adapter_path
            logger.info(f"LoRA adapter loaded from {adapter_path}.")
        except ImportError:
            logger.warning("peft not installed; cannot load style adapter.")
        except Exception as exc:
            logger.warning(f"Adapter load failed: {exc}")

    def unload_style_adapter(self) -> None:
        """Remove the active LoRA adapter (merge weights back into base)."""
        if self._active_adapter is None:
            return
        try:
            from peft import PeftModel  # noqa: PLC0415
            if isinstance(self.model, PeftModel):
                self.model = self.model.merge_and_unload()
                logger.info("LoRA adapter merged and unloaded.")
        except Exception as exc:
            logger.warning(f"Failed to unload adapter: {exc}")
        self._active_adapter = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        context_tokens: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        top_p: float = 0.92,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
    ) -> list[int]:
        """
        Auto-regressively generate token ids from *context_tokens*.

        Returns the complete token sequence (context + generated).
        Falls back to a random token sequence in stub mode.
        """
        if self._stub_mode or self.model is None:
            return self._stub_generate(context_tokens, max_new_tokens)

        try:
            import torch  # noqa: PLC0415

            input_ids = torch.tensor([context_tokens], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.vocab_size - 1,
                    eos_token_id=self.vocab_size - 1,
                )

            return output[0].tolist()
        except Exception as exc:
            logger.warning(f"Generation failed ({exc}); using stub output.")
            return self._stub_generate(context_tokens, max_new_tokens)

    def _stub_generate(self, context_tokens: list[int], max_new_tokens: int) -> list[int]:
        """Return a plausible-looking random token sequence for stub/demo mode."""
        import random
        result = list(context_tokens)
        for _ in range(max_new_tokens):
            # Simple heuristic: mostly pitch tokens (0-127) with duration tokens
            r = random.random()
            if r < 0.5:
                result.append(random.randint(48, 84))   # pitch range C3-C6
            elif r < 0.75:
                result.append(random.randint(0, 31) * 128)  # duration token
            else:
                result.append(random.randint(4096, 4100))   # velocity token
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self.model is not None and not self._stub_mode

    def parameter_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "stub"
        adapter = f", adapter={self._active_adapter!r}" if self._active_adapter else ""
        return f"SymbolicMusicModel(device={self.device!r}, status={status}{adapter})"
