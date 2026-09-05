"""Named bundles of every stage's config. The Advanced panel exposes the
individual knobs; a preset just seeds them."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .analysis import AnalysisConfig
from .fuse import FuseConfig
from .refine import RefineConfig
from .render import RenderConfig
from .score import ScoreConfig


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    analysis: AnalysisConfig
    refine: RefineConfig
    score: ScoreConfig
    render: RenderConfig


PRESETS: dict[str, Preset] = {
    "Balanced": Preset(
        "Balanced",
        "Every source votes, long holds, review sorted by suspicion.",
        AnalysisConfig(), RefineConfig(), ScoreConfig(auto_disable_above=0.65),
        RenderConfig()),
    "Max Privacy": Preset(
        "Max Privacy",
        "Lowest spawn bar, longest holds, nothing auto-disabled, bigger "
        "and harder blur.",
        AnalysisConfig(spawn_conf=0.25, sustain_conf=0.08, max_age_s=3.5,
                       headdet_floor=0.08, scrfd_floor=0.25, scrfd_size=1280),
        RefineConfig(bridge_gap_s=3.5, extend_s=0.5, min_hit_ratio=0.10,
                     min_track_s=0.12),
        ScoreConfig(auto_disable_above=1.01),
        RenderConfig(mask_pad=0.30, motion_lead=0.8,
                     blur_layers=(("gaussian", 99), ("pixelate", 16)))),
    "Fast": Preset(
        "Fast",
        "Every 3rd frame, fewer rotation passes; the offline stages fill in "
        "between. Trades recall on brief appearances for ~5× speed.",
        AnalysisConfig(stride=3, headdet_rots=(0, 180), wb_rots=(0,),
                       scrfd_rots=(0,)),
        RefineConfig(bridge_gap_s=3.0, extend_s=0.4),
        ScoreConfig(auto_disable_above=0.65), RenderConfig()),
    "Strict": Preset(
        "Strict",
        "Higher spawn bar and more auto-disabling for footage where false "
        "blurs matter.",
        AnalysisConfig(spawn_conf=0.45, min_hits=3,
                       fuse=FuseConfig(bodypart_weight=0.3)),
        RefineConfig(min_hits=5, min_track_s=0.40, min_top_score=0.30),
        ScoreConfig(auto_disable_above=0.50),
        RenderConfig(mask_pad=0.15, motion_lead=0.4)),
}

DEFAULT_PRESET = "Balanced"


def with_stride(p: Preset, stride: int) -> Preset:
    return replace(p, analysis=replace(p.analysis, stride=max(1, int(stride))))
