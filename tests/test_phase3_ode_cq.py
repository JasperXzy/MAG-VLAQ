import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch

from network_mm.fuse_block_toshallow import FuseBlockToShallow
from network_mm.mm import MM
from network_mm.ode_cq import DeltaQ
from network_mm.vlaq import VLAQ


def test_delta_q_zero_up_outputs_zero_but_trains_up_projection():
    torch.manual_seed(0)
    delta_q = DeltaQ(C=8, S=4, D=6, r=3, alpha_init=1.0, alpha_learn=False)
    e_fuse = torch.randn(2, 8)
    q_bias = delta_q(e_fuse)

    assert q_bias.shape == (2, 4, 6)
    assert torch.allclose(q_bias, torch.zeros_like(q_bias))
    assert torch.allclose(delta_q.w_up_q.weight, torch.zeros_like(delta_q.w_up_q.weight))

    target = torch.randn_like(q_bias)
    loss = (q_bias * target).sum()
    loss.backward()
    assert delta_q.w_up_q.weight.grad is not None
    assert delta_q.w_up_q.weight.grad.abs().sum() > 0


def test_vlaq_zero_q_bias_matches_static_queries():
    torch.manual_seed(0)
    vlaq = VLAQ(
        n_queries=4,
        query_dim=8,
        token_dim=8,
        out_dim=16,
        dropout=0.0,
        q_init="orthogonal",
    )
    tokens = torch.randn(2, 5, 8)
    q_bias = torch.zeros(2, 4, 8)

    static_out = vlaq(tokens, q_bias=None)
    biased_out = vlaq(tokens, q_bias=q_bias)

    assert torch.allclose(static_out, biased_out, atol=1e-6)


def test_fuse_summary_accepts_charted_2d_tokens():
    block = object.__new__(FuseBlockToShallow)
    block.args = SimpleNamespace(fuse_summary_mode="mean")
    tokens = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    summary = FuseBlockToShallow.per_scale_summary(block, tokens, "2d", 0)

    assert summary.shape == (2, 4)
    assert torch.allclose(summary, tokens.mean(dim=1))


class _DummyImageFE(nn.Module):
    last_dim = 1024

    def __init__(self, *args, **kwargs):
        super().__init__()


class _DummyUtoniaFE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ptv3 = nn.Linear(1, 1)
        self.projs = nn.ModuleList([nn.Linear(1, 1)])


def _mm_args(use_ode_cq=True):
    return SimpleNamespace(
        mm_imgfe_dim=1024,
        mm_voxfe_planes="64_128_256",
        mm_bevfe_planes="64_128_256",
        mm_stg2fuse_dim=512,
        mm_bevfe_dim=256,
        mm_voxfe_dim=256,
        vlaq_token_dim=256,
        vlaq_n_queries=64,
        vlaq_query_dim=64,
        ode_cq_rank=16,
        ode_cq_alpha_init=0.0,
        ode_cq_alpha_learn=False,
        ode_cq_bias_scale=1.0,
        ode_cq_max_ratio=0.0,
        use_ode_cq=use_ode_cq,
        final_type="vlaq_only",
        output_type=["image", "vox", "shallow"],
        features_dim=512,
        image_weight=1.0,
        image_learnweight=False,
        vox_weight=1.0,
        vox_learnweight=False,
        shallow_weight=1.0,
        shallow_learnweight=False,
        imagevoxorg_weight=0.0,
        imagevoxorg_learnweight=False,
        shalloworg_weight=1.0,
        shalloworg_learnweight=False,
        stg2imagevox_weight=0.1,
        stg2imagevox_learnweight=False,
        stg2fuse_weight=0.0,
        stg2fuse_learnweight=False,
        diff_type="fcode@relu",
        diff_direction="backward",
        odeint_method="euler",
        odeint_size=0.1,
        tol=1e-3,
    )


def test_ode_cq_reuses_static_last_voxel_chart():
    with patch("network_mm.mm.ImageFE", _DummyImageFE), patch(
        "network_mm.mm.UtoniaFE", _DummyUtoniaFE
    ):
        model = MM(args=_mm_args(use_ode_cq=True))

    assert model.chart_vox_l[-1] is model.chart_vox


def test_ode_cq_extra_modules_do_not_shift_static_initialization():
    with patch("network_mm.mm.ImageFE", _DummyImageFE), patch(
        "network_mm.mm.UtoniaFE", _DummyUtoniaFE
    ):
        torch.manual_seed(123)
        static_model = MM(args=_mm_args(use_ode_cq=False))
        static_weights = {
            "chart_img": static_model.chart_img.proj.weight.detach().clone(),
            "chart_vox": static_model.chart_vox.proj.weight.detach().clone(),
            "image_proj": static_model.image_proj.seq[0].weight.detach().clone(),
            "vox_proj": static_model.vox_proj.seq[0].weight.detach().clone(),
        }

        torch.manual_seed(123)
        odecq_model = MM(args=_mm_args(use_ode_cq=True))
        odecq_weights = {
            "chart_img": odecq_model.chart_img.proj.weight.detach().clone(),
            "chart_vox": odecq_model.chart_vox.proj.weight.detach().clone(),
            "image_proj": odecq_model.image_proj.seq[0].weight.detach().clone(),
            "vox_proj": odecq_model.vox_proj.seq[0].weight.detach().clone(),
        }

    for name, weight in static_weights.items():
        assert torch.allclose(weight, odecq_weights[name]), name


def test_shared_vlaq_default_uses_global_rng_sequence():
    from lit.module import SCAModule

    args = SimpleNamespace(
        features_dim=512,
        vlaq_n_queries=64,
        vlaq_query_dim=64,
        vlaq_token_dim=256,
        vlaq_out_dim=512,
        vlaq_dropout=0.0,
        vlaq_q_init="orthogonal",
        vlaq_init_seed=None,
    )
    torch.manual_seed(321)
    expected = VLAQ(
        n_queries=args.vlaq_n_queries,
        query_dim=args.vlaq_query_dim,
        token_dim=args.vlaq_token_dim,
        out_dim=args.vlaq_out_dim,
        dropout=args.vlaq_dropout,
        q_init=args.vlaq_q_init,
    )

    torch.manual_seed(321)
    module = object.__new__(SCAModule)
    nn.Module.__init__(module)
    module.args = args
    module.modelq = SimpleNamespace()
    module.model = SimpleNamespace()
    SCAModule._init_shared_vlaq(module)

    for expected_param, actual_param in zip(
        expected.parameters(), module.shared_vlaq.parameters()
    ):
        assert torch.allclose(expected_param, actual_param)
