"""
adapters/model_wrapper.py
==========================
HuggingFace transformers + PEFT wrapper for the symbolic music generation model.

The base model is GPT-2-small adapted for MIDI token vocabularies.
Style packs are loaded as PEFT LoRA adapters on top of the base model.

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


class SymbolicMusicModel:
    """
    Wrapper around a GPT-2-small model adapted for MIDI token sequences.

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
        Load the base GPT-2-small model from *model_name_or_path*.

        If the path does not exist, falls back to loading the public
        'gpt2' checkpoint from HuggingFace Hub and resizing its token
        embedding to *vocab_size*.  In offline mode without weights,
        activates stub mode.
        """
        if self.model is not None and not force_reload:
            logger.debug("Base model already loaded; skipping reload.")
            return

        path = Path(model_name_or_path)
        logger.info(f"Loading base model from '{model_name_or_path}' on {self.device}…")

        try:
            from transformers import GPT2Config, GPT2LMHeadModel  # noqa: PLC0415
            import torch  # noqa: PLC0415

            if path.exists() and (path / "config.json").exists():
                self.model = GPT2LMHeadModel.from_pretrained(str(path))
                logger.info(f"Loaded base model from local path {path}.")
            else:
                logger.warning(
                    f"Local model not found at {path}. "
                    "Falling back to 'gpt2' from HuggingFace Hub."
                )
                self.hf_config = GPT2Config.from_pretrained("gpt2")
                self.hf_config.vocab_size = self.vocab_size
                self.model = GPT2LMHeadModel(self.hf_config)
                logger.info("Initialised GPT-2-small with random weights (no pretrained checkpoint).")

            self.model = self.model.to(self.device)
            self.model.eval()
            self._stub_mode = False
            logger.info(
                f"Base model ready. Parameters: "
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
