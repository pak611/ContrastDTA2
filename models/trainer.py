# trainer.py
import torch
from torch import nn
from contextlib import nullcontext
from typing import Optional, Callable, Dict, Any

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader=None,
        optimizer: Optional[torch.optim.Optimizer]=None,
        criterion: Optional[Callable]=None,          # e.g., nn.MSELoss()
        scheduler: Optional[Any]=None,
        device: str | torch.device = "cpu",
        log_interval: int = 50,
        use_amp: bool = True,
        grad_clip: Optional[float] = None,           # e.g., 1.0
        accum_steps: int = 1,                        # gradient accumulation
        metric_fn: Optional[Callable]=None,          # optional extra metric fn(pred, y) -> dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        self.criterion = criterion or nn.MSELoss()
        self.log_interval = log_interval
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = grad_clip
        self.accum_steps = max(1, accum_steps)
        self.metric_fn = metric_fn
        self.current_epoch = None

    def _forward_and_loss(self, batch):
        """Support models that return a tensor or a dict with 'pred' and/or 'loss'."""
        out = self.model(batch)  # expect HeteroData batch
        if isinstance(out, dict):
            pred = out.get("pred", None)
            loss = out.get("loss", None)
            if loss is None:
                # compute with criterion if 'pred' present
                if pred is None:
                    raise ValueError("Model dict output must contain 'pred' or 'loss'.")
                loss = self.criterion(pred.view(-1), batch.y.view(-1))
        else:
            pred = out
            loss = self.criterion(pred.view(-1), batch.y.view(-1))
        return pred, loss

    # removed contrastive helpers

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
        for i, batch in enumerate(self.train_loader, start=1):
            batch = batch.to(self.device)

            with autocast_ctx():
                pred, loss = self._forward_and_loss(batch)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if i % self.accum_steps == 0:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    # step per-optimizer-step; change to per-epoch outside if you prefer
                    self.scheduler.step()

            total_loss += float(loss.detach().cpu()) * self.accum_steps
            if (i % self.log_interval) == 0:
                print(f"train iter {i}: loss={float(loss)*self.accum_steps:.4f}")

        avg = total_loss / max(1, len(self.train_loader))
        print(f"train avg loss: {avg:.4f}")
        return avg

    @torch.no_grad()
    def validate(self) -> Optional[Dict[str, float]]:
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        metric_sums: Dict[str, float] = {}

        autocast_ctx = torch.cuda.amp.autocast if self.use_amp else nullcontext
        for batch in self.val_loader:
            batch = batch.to(self.device)
            with autocast_ctx():
                pred, loss = self._forward_and_loss(batch)
            total_loss += float(loss.detach().cpu())
            n_batches += 1

            if self.metric_fn is not None and pred is not None:
                metrics = self.metric_fn(pred.detach().cpu().view(-1), batch.y.detach().cpu().view(-1))
                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

        avg_loss = total_loss / max(1, n_batches)
        result = {"val_loss": avg_loss}
        if self.metric_fn is not None and n_batches > 0:
            for k, s in metric_sums.items():
                result[k] = s / n_batches
        print("val  metrics:", ", ".join(f"{k}={v:.4f}" for k, v in result.items()))
        return result

    def fit(self, epochs: int):
        history = []
        for ep in range(1, epochs + 1):
            print(f"Epoch {ep}/{epochs}")
            self.current_epoch = ep
            tr = self.train_epoch()
            vr = self.validate()
            history.append({"epoch": ep, "train_loss": tr, **(vr or {})})
        return history

    @torch.no_grad()
    def save_interaction_maps(self, loader, out_dir: str | None, epoch: int, limit: int = 8):
        """Collect and save up to `limit` interaction maps from the first batches of `loader`.

        Saves .pt files under out_dir/epoch_{epoch}/map_{k}.pt containing a dict:
          { 'map': Tensor[Nl,Np], 'shape': (Nl,Np) }
        If the model does not support returning maps, this is a no-op.
        """
        if out_dir is None:
            return
        os = __import__('os')
        from pathlib import Path
        self.model.eval()
        saved = 0
        out_base = Path(out_dir) / f"epoch_{epoch}"
        out_base.mkdir(parents=True, exist_ok=True)
        for batch in loader:
            batch = batch.to(self.device)
            try:
                out = self.model(batch, return_maps=True)
                if not isinstance(out, dict) or 'maps' not in out:
                    # model does not return maps; stop trying
                    return
                maps = out['maps']  # list of [Nl_i, Np_i]
                for m in maps:
                    if saved >= limit:
                        return
                    m_cpu = m.detach().cpu()
                    torch.save({"map": m_cpu, "shape": tuple(m_cpu.shape)}, out_base / f"map_{saved:03d}.pt")
                    saved += 1
            except TypeError:
                # model forward may not accept return_maps
                return
            except Exception:
                # be robust: don't crash training if map saving fails
                return
