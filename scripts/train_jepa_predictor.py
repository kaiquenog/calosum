#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainPair:
    context_turns: list[str]
    good_response: str
    bad_response: str | None = None


class EmbeddingEncoder:
    def __init__(
        self,
        model_name: str,
        embed_dim: int = 384,
        use_sentence_transformers: bool = True,
    ) -> None:
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.use_sentence_transformers = use_sentence_transformers
        self._embedder: Any | None = None

    def encode(self, text: str) -> list[float]:
        if not self.use_sentence_transformers:
            return self._lexical_vector(text)
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.model_name)
            vector = self._embedder.encode([text], normalize_embeddings=True)[0].tolist()
            return self._normalize_size(vector)
        except Exception:
            return self._lexical_vector(text)

    def _normalize_size(self, vector: list[float]) -> list[float]:
        if len(vector) > self.embed_dim:
            vector = vector[: self.embed_dim]
        elif len(vector) < self.embed_dim:
            vector = vector + ([0.0] * (self.embed_dim - len(vector)))
        return self._l2_normalize(vector)

    def _lexical_vector(self, text: str) -> list[float]:
        tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split() if token.strip(".,;:!?()[]{}\"'")] or ["silence"]
        vector = [0.0] * self.embed_dim
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index:index + 4]
                pos = int.from_bytes(chunk[:2], "big") % self.embed_dim
                sign = 1.0 if chunk[2] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[3] / 255.0)
                vector[pos] += sign * magnitude
        return self._l2_normalize(vector)

    def _l2_normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0] * len(vector)
        return [float(value / norm) for value in vector]


def _load_pairs(path: Path) -> list[TrainPair]:
    pairs: list[TrainPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            context_turns = row.get("context_turns") or row.get("context") or []
            if isinstance(context_turns, str):
                context_turns = [context_turns]
            good_response = row.get("good_response") or row.get("response") or ""
            if not context_turns or not good_response:
                continue
            bad_response = row.get("bad_response")
            pairs.append(
                TrainPair(
                    context_turns=[str(turn) for turn in context_turns if str(turn).strip()],
                    good_response=str(good_response),
                    bad_response=str(bad_response) if bad_response is not None else None,
                )
            )
    return pairs


def _pad_context(context_vectors: list[list[float]], size: int, embed_dim: int) -> list[list[float]]:
    context_vectors = context_vectors[-size:]
    pad = [[0.0] * embed_dim for _ in range(max(0, size - len(context_vectors)))]
    return pad + context_vectors


def _vector_to_tensor(torch: Any, vectors: list[list[float]], device: str):
    return torch.tensor(vectors, dtype=torch.float32, device=device)


def _pairs_to_tensors(
    pairs: list[TrainPair],
    *,
    encoder: EmbeddingEncoder,
    max_turns: int,
    embed_dim: int,
    torch: Any,
    device: str,
):
    ctx_rows: list[list[list[float]]] = []
    good_rows: list[list[float]] = []
    bad_rows: list[list[float]] = []

    for pair in pairs:
        context_vectors = [encoder.encode(turn) for turn in pair.context_turns]
        context_matrix = _pad_context(context_vectors, max_turns, embed_dim)
        good = encoder.encode(pair.good_response)
        bad = encoder.encode(pair.bad_response or "Resposta ruim fora de contexto")
        ctx_rows.append(context_matrix)
        good_rows.append(good)
        bad_rows.append(bad)

    return (
        _vector_to_tensor(torch, ctx_rows, device),
        _vector_to_tensor(torch, good_rows, device),
        _vector_to_tensor(torch, bad_rows, device),
    )


def _build_predictor(torch: Any, embed_dim: int, hidden: int, dropout: float):
    from torch import nn
    import torch.nn.functional as F

    class JEPAPredictor(nn.Module):
        def __init__(self, embed_dim: int = 384, hidden: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed_dim * 3, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, embed_dim),
            )

        def forward(self, ctx_embeds):
            return F.normalize(self.net(ctx_embeds.flatten(1)), dim=-1)

    return JEPAPredictor(embed_dim=embed_dim, hidden=hidden)


def _make_batches(torch: Any, *tensors, batch_size: int):
    total = tensors[0].shape[0]
    indices = torch.randperm(total, device=tensors[0].device)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        idx = indices[start:end]
        yield tuple(t.index_select(0, idx) for t in tensors)


def _ranking_accuracy(torch: Any, model: Any, ctx: Any, good: Any, bad: Any, batch_size: int) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for c_batch, g_batch, b_batch in _make_batches(torch, ctx, good, bad, batch_size=batch_size):
            pred = model(c_batch)
            good_sim = (pred * g_batch).sum(dim=-1)
            bad_sim = (pred * b_batch).sum(dim=-1)
            correct += int((good_sim > bad_sim).sum().item())
            total += int(pred.shape[0])
    return float(correct / max(1, total))


def _evaluate_loss(torch: Any, model: Any, ctx: Any, good: Any, bad: Any, batch_size: int) -> float:
    from torch.nn.functional import cosine_similarity, triplet_margin_loss

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for c_batch, g_batch, b_batch in _make_batches(torch, ctx, good, bad, batch_size=batch_size):
            pred = model(c_batch)
            cosine = 1.0 - cosine_similarity(pred, g_batch).mean()
            triplet = triplet_margin_loss(pred, g_batch, b_batch, margin=0.2)
            loss = cosine + (0.1 * triplet)
            losses.append(float(loss.item()))
    if not losses:
        return 0.0
    return sum(losses) / len(losses)


def _uncertainty_surprise_alignment(torch: Any, model: Any, ctx: Any, bad: Any, n_samples: int = 10) -> float:
    if ctx.shape[0] == 0:
        return 0.0
    model.train()
    with torch.no_grad():
        preds = torch.stack([model(ctx) for _ in range(max(1, n_samples))], dim=0)
        mean = preds.mean(dim=0)
        variance = preds.var(dim=0).mean(dim=-1)
        surprise = (1.0 - (mean * bad).sum(dim=-1)) / 2.0

    var_threshold = torch.quantile(variance, 0.7)
    surprise_threshold = torch.quantile(surprise, 0.7)
    high_uncertainty = variance >= var_threshold
    high_surprise = surprise >= surprise_threshold
    aligned = (high_uncertainty == high_surprise).float().mean()
    model.eval()
    return float(aligned.item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina preditor JEPA textual fase 2")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("adapters/jepa_predictor/v1.0"))
    parser.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--sources", default="OpenAssistant/oasst2,Anthropic/hh-rlhf,HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--uncertainty-samples", type=int, default=10)
    parser.add_argument("--min-train", type=int, default=50_000)
    parser.add_argument("--min-val", type=int, default=5_000)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--force-lexical-encoder", action="store_true")
    parser.add_argument("--version", default="1.0")
    args = parser.parse_args()

    if not args.train_jsonl.exists() or not args.val_jsonl.exists():
        raise FileNotFoundError("train/val jsonl nao encontrados")

    train_pairs = _load_pairs(args.train_jsonl)
    val_pairs = _load_pairs(args.val_jsonl)

    if not args.allow_small_dataset:
        if len(train_pairs) < args.min_train:
            raise RuntimeError(f"train set insuficiente: {len(train_pairs)} < {args.min_train}")
        if len(val_pairs) < args.min_val:
            raise RuntimeError(f"val set insuficiente: {len(val_pairs)} < {args.min_val}")

    try:
        import torch
        from torch.nn.functional import cosine_similarity, triplet_margin_loss
    except Exception as exc:
        raise RuntimeError("PyTorch e obrigatorio para treinar o preditor JEPA") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EmbeddingEncoder(
        model_name=args.encoder_model,
        embed_dim=args.embed_dim,
        use_sentence_transformers=not args.force_lexical_encoder,
    )

    train_ctx, train_good, train_bad = _pairs_to_tensors(
        train_pairs,
        encoder=encoder,
        max_turns=args.max_turns,
        embed_dim=args.embed_dim,
        torch=torch,
        device=device,
    )
    val_ctx, val_good, val_bad = _pairs_to_tensors(
        val_pairs,
        encoder=encoder,
        max_turns=args.max_turns,
        embed_dim=args.embed_dim,
        torch=torch,
        device=device,
    )

    model = _build_predictor(torch, args.embed_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, args.warmup_steps)))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        for c_batch, g_batch, b_batch in _make_batches(
            torch,
            train_ctx,
            train_good,
            train_bad,
            batch_size=args.batch_size,
        ):
            optimizer.zero_grad(set_to_none=True)
            pred = model(c_batch)
            cosine = 1.0 - cosine_similarity(pred, g_batch).mean()
            triplet = triplet_margin_loss(pred, g_batch, b_batch, margin=0.2)
            loss = cosine + (0.1 * triplet)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.item()))

        val_loss = _evaluate_loss(torch, model, val_ctx, val_good, val_bad, args.batch_size)
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early_stopping epoch={epoch + 1} patience={args.patience}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ranking_acc = _ranking_accuracy(torch, model, val_ctx, val_good, val_bad, args.batch_size)
    uncertainty_alignment = _uncertainty_surprise_alignment(
        torch,
        model,
        val_ctx,
        val_bad,
        n_samples=args.uncertainty_samples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "predictor.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "embed_dim": args.embed_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "max_turns": args.max_turns,
            },
            "metrics": {
                "ranking_accuracy": ranking_acc,
                "uncertainty_surprise_alignment": uncertainty_alignment,
                "best_val_loss": best_val_loss,
            },
            "trained_at": datetime.now(UTC).isoformat(),
        },
        checkpoint_path,
    )

    metadata = {
        "version": args.version,
        "model_name": f"trained-jepa-v{args.version}",
        "sources": [src.strip() for src in args.sources.split(",") if src.strip()],
        "encoder": {
            "name": args.encoder_model,
            "frozen": True,
            "backend": "lexical" if args.force_lexical_encoder else "sentence_transformers",
        },
        "training": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
        },
        "metrics": {
            "ranking_accuracy": ranking_acc,
            "uncertainty_surprise_alignment": uncertainty_alignment,
            "best_val_loss": best_val_loss,
            "target_gate": {
                "ranking_accuracy_gte": 0.75,
                "uncertainty_alignment_gte": 0.70,
            },
            "gate_passed": ranking_acc >= 0.75 and uncertainty_alignment >= 0.70,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "train_input": str(args.train_jsonl),
            "val_input": str(args.val_jsonl),
        },
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    metadata_path = args.output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_metadata={metadata_path}")
    print(f"ranking_accuracy={ranking_acc:.4f}")
    print(f"uncertainty_surprise_alignment={uncertainty_alignment:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
