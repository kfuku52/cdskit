"""Loss components whose normalization is independent of evaluation batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class TargetPLoss:
    type_sum: Any
    type_weight: Any
    cleavage_sum: Any
    count: int

    def mean(self) -> Any:
        return self.type_sum / self.type_weight + self.cleavage_sum / self.count


@dataclass
class TargetPLossTotals:
    type_sum: float = 0.0
    type_weight: float = 0.0
    cleavage_sum: float = 0.0
    count: int = 0

    def add(self, loss: TargetPLoss) -> None:
        self.type_sum += float(loss.type_sum.detach().cpu())
        self.type_weight += float(loss.type_weight)
        self.cleavage_sum += float(loss.cleavage_sum.detach().cpu())
        self.count += loss.count

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        if self.type_weight <= 0:
            raise ValueError("TargetP loss requires a positive total class weight.")
        return self.type_sum / self.type_weight + self.cleavage_sum / self.count


def targetp_loss_components(
    torch: Any,
    outputs: Mapping[str, Any],
    y_type: Any,
    y_cs: Any,
    signal_class_to_head: Mapping[int, int],
    type_weight: Any = None,
    cleavage_loss_weight: float = 1.0,
) -> TargetPLoss:
    from torch.nn import functional as F

    labels = y_type.long()
    count = int(labels.shape[0])
    type_sum = F.cross_entropy(
        outputs["type_logits"], labels, weight=type_weight, reduction="sum"
    )
    denominator = count if type_weight is None else type_weight[labels].sum()
    cleavage_sum = type_sum.new_zeros(())
    if float(cleavage_loss_weight) > 0.0:
        positions = torch.argmax(y_cs.long(), dim=1)
        for class_idx, head_idx in signal_class_to_head.items():
            mask = labels == int(class_idx)
            if torch.any(mask):
                cleavage_sum = cleavage_sum + float(
                    cleavage_loss_weight
                ) * F.cross_entropy(
                    outputs["attention_logits"][mask, :, int(head_idx)],
                    positions[mask],
                    reduction="sum",
                )
    return TargetPLoss(type_sum, denominator, cleavage_sum, count)
