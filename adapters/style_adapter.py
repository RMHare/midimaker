"""
adapters/style_adapter.py
==========================
PEFT LoRA adapter creation, training, saving, and loading.

Wraps HuggingFace PEFT to provide a clean interface for the rest of the
MidiMaker codebase.  Falls back gracefully when peft / torch are absent.

Usage::

    adapter = StyleAdapter(device="cuda")
    adapter.create_adapter("assets/base_model", rank=8, alpha=32)
    adapter.train_adapter(token_sequences, epochs=10)
    adapter.save_adapter(Path("styles/jazz_piano/adapter"))

    # Later:
    adapter2 = StyleAdapter()
    adapter2.load_adapter(Path("styles/jazz_piano/adapter"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger


class StyleAdapter:
    """
    Manages a PEFT LoRA adapter for the symbolic music generation model.

    Attributes
    ----------
    model : peft.PeftModel or None
        The LoRA-adapted model (set after create_adapter or load_adapter).
    device : str
        Compute device.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        from core.utils import get_device
        self.device = device or get_device()
        self.model: Any = None
        self._base_model: Any = None
        self._rank = 8
        self._alpha = 32
        self._stub_mode = False

    # ------------------------------------------------------------------
    # Adapter creation
    # ------------------------------------------------------------------

    def create_adapter(
        self,
        base_model_path: str,
        rank: int = 8,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> Any:
        """
        Wrap the base model with a LoRA adapter.

        Parameters
        ----------
        base_model_path : str
            Path to the base GPT-2 checkpoint (or a HuggingFace model name).
        rank : int
            LoRA rank (r).
        alpha : int
            LoRA alpha scaling factor.
        dropout : float
            Dropout on the LoRA projection matrices.
        target_modules : list[str], optional
            Which attention projection names to adapt.
            Defaults to ['c_attn'] (GPT-2 combined QKV projection).

        Returns the adapted model.
        """
        self._rank = rank
        self._alpha = alpha
        target_modules = target_modules or ["c_attn"]

        try:
            import torch  # noqa: PLC0415
            from transformers import GPT2LMHeadModel  # noqa: PLC0415
            from peft import LoraConfig, get_peft_model, TaskType  # noqa: PLC0415

            path = Path(base_model_path)
            if path.exists() and (path / "config.json").exists():
                base = GPT2LMHeadModel.from_pretrained(str(path))
            else:
                logger.warning(
                    f"Base model not found at {base_model_path}; "
                    "using random-weight GPT-2-small."
                )
                from transformers import GPT2Config  # noqa: PLC0415
                cfg = GPT2Config()
                base = GPT2LMHeadModel(cfg)

            base = base.to(self.device)
            self._base_model = base

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
            )
            self.model = get_peft_model(base, lora_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                f"LoRA adapter created: rank={rank}, alpha={alpha}, "
                f"trainable params={trainable / 1e3:.1f}K"
            )
            return self.model
        except ImportError as exc:
            logger.warning(f"PEFT/torch not available ({exc}); activating stub mode.")
            self._stub_mode = True
            return None
        except Exception as exc:
            logger.warning(f"create_adapter failed ({exc}); activating stub mode.")
            self._stub_mode = True
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_adapter(
        self,
        token_sequences: list[list[int]],
        epochs: int = 10,
        learning_rate: float = 2e-4,
        batch_size: int = 8,
        max_seq_len: int = 512,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """
        Fine-tune the LoRA adapter on the given token sequences.

        Uses HuggingFace Trainer in its simplest form for portability.
        Falls back to a stub (no-op) if torch is not available.
        """
        if self._stub_mode or self.model is None:
            logger.info("Stub mode: skipping adapter training.")
            _cb(progress_callback, epochs, epochs, "Training skipped (stub mode).")
            return

        logger.info(
            f"Training LoRA adapter: epochs={epochs}, lr={learning_rate:.1e}, "
            f"sequences={len(token_sequences)}"
        )

        try:
            import torch  # noqa: PLC0415
            from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415
            from torch.optim import AdamW  # noqa: PLC0415

            self.model.train()

            # Build dataset: each sequence padded/truncated to max_seq_len
            sequences_t = []
            for seq in token_sequences:
                if len(seq) > max_seq_len:
                    seq = seq[:max_seq_len]
                elif len(seq) < max_seq_len:
                    seq = seq + [0] * (max_seq_len - len(seq))
                sequences_t.append(seq)

            if not sequences_t:
                logger.warning("No training sequences; skipping.")
                return

            input_ids = torch.tensor(sequences_t, dtype=torch.long)
            dataset = TensorDataset(input_ids)
            loader = DataLoader(dataset, batch_size=min(batch_size, len(sequences_t)), shuffle=True)
            optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=learning_rate,
            )

            total_steps = epochs * len(loader)
            step = 0
            for epoch in range(epochs):
                epoch_loss = 0.0
                for (batch_input,) in loader:
                    batch_input = batch_input.to(self.device)
                    outputs = self.model(input_ids=batch_input, labels=batch_input)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    step += 1

                avg_loss = epoch_loss / max(len(loader), 1)
                msg = f"Epoch {epoch + 1}/{epochs} — loss={avg_loss:.4f}"
                logger.debug(msg)
                _cb(progress_callback, epoch + 1, epochs, msg)

            self.model.eval()
            logger.info("LoRA adapter training complete.")
        except Exception as exc:
            logger.warning(f"Training failed: {exc}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_adapter(self, path: Path) -> None:
        """Save the LoRA adapter weights to *path* using PEFT's safetensors format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._stub_mode or self.model is None:
            # Write a stub config so the directory is non-empty
            stub_config = {
                "base_model_name_or_path": "gpt2",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": self._rank,
                "lora_alpha": self._alpha,
                "target_modules": ["c_attn"],
                "stub": True,
            }
            (path / "adapter_config.json").write_text(
                json.dumps(stub_config, indent=2), encoding="utf-8"
            )
            logger.info(f"Stub adapter config saved to {path}.")
            return

        try:
            self.model.save_pretrained(str(path))
            logger.info(f"LoRA adapter saved to {path}.")
        except Exception as exc:
            logger.warning(f"save_adapter failed: {exc}")

    def load_adapter(self, path: Path) -> None:
        """Load a previously saved LoRA adapter."""
        path = Path(path)
        config_file = path / "adapter_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {path}")

        config = json.loads(config_file.read_text(encoding="utf-8"))
        if config.get("stub"):
            logger.info("Loading stub adapter (no actual weights); stub mode active.")
            self._stub_mode = True
            return

        try:
            from peft import PeftModel  # noqa: PLC0415
            from transformers import GPT2LMHeadModel  # noqa: PLC0415

            base_model_name = config.get("base_model_name_or_path", "gpt2")
            base = GPT2LMHeadModel.from_pretrained(base_model_name)
            self.model = PeftModel.from_pretrained(base, str(path))
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"LoRA adapter loaded from {path}.")
        except ImportError:
            logger.warning("peft/transformers not installed; stub mode.")
            self._stub_mode = True
        except Exception as exc:
            logger.warning(f"load_adapter failed: {exc}; stub mode.")
            self._stub_mode = True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self.model is not None and not self._stub_mode

    def trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        status = "trained" if self.is_trained() else "stub"
        return (
            f"StyleAdapter(device={self.device!r}, rank={self._rank}, "
            f"alpha={self._alpha}, status={status})"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _cb(
    callback: Optional[Callable[[int, int, str], None]],
    step: int,
    total: int,
    msg: str,
) -> None:
    if callback:
        try:
            callback(step, total, msg)
        except Exception:
            pass
