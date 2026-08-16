"""CPU smoke test for the public Reffuser architecture components."""

import torch

from baselines.PatchTST.arch import PatchTST
from baselines.Reffuser.arch.schedule import DiffusionSchedule
from baselines.Reffuser.arch.time_schedule import TimeVectorNetwork


def main() -> None:
    torch.manual_seed(0)
    batch_size, history_length, horizon, nodes = 2, 24, 24, 7
    history = torch.randn(batch_size, history_length, nodes, 1)
    future = torch.randn(batch_size, horizon, nodes, 1)

    actor = PatchTST(
        enc_in=nodes,
        seq_len=history_length,
        pred_len=horizon,
        e_layers=1,
        n_heads=1,
        d_model=16,
        d_ff=32,
        dropout=0.0,
        fc_dropout=0.0,
        head_dropout=0.0,
        patch_len=4,
        stride=2,
        individual=0,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=5,
    )
    prediction = actor(history, future, batch_seen=0, epoch=0, train=False)
    assert prediction.shape == (batch_size, horizon, nodes, 1)
    assert torch.isfinite(prediction).all()

    segment_count = 1
    time_schedule = TimeVectorNetwork(
        input_dim=(history_length + horizon) * nodes,
        output_dim=segment_count,
        hidden_dim=16,
        max_time=1000,
    )
    positions = time_schedule(history, future)
    assert positions.shape == (segment_count,)
    assert torch.isfinite(positions).all()

    noise_schedule = DiffusionSchedule()
    noise = noise_schedule.get_sigma_squared(positions.unsqueeze(-1))
    assert torch.isfinite(noise).all()
    print("Public model-component smoke test passed.")


if __name__ == "__main__":
    main()
