"""Union-detection privacy pipeline: detect (every head-ish claim from every
source) → track (long memory) → refine offline (bridge, extend, smooth) →
score (suspicion, for review ranking) → render.

Package map:
  geom      pure-numpy box helpers shared by every stage
  types     Src flags, Tracklet (step space), Track (frame space + review meta)
  fuse      per-frame union of detector outputs into head candidates
  tracker   Kalman + BYTE + Hungarian online tracker over candidates
  record    per-frame TrackObs → contiguous Tracklets
  refine    offline: trim, soft prune, bridge, extend, smooth, upsample
  score     suspicion components + optional magnified re-detection
  render    per-frame blur table (velocity-aware) + export loop
  analysis  decode loop driving fuse → tracker → record on a stride
  project   sidecar persistence (analysis cache, decisions, manual tracks)
  propagate manual box propagation (candidate snapping + template match)
"""
