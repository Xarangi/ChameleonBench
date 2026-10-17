"""In-context activation evasion with defender feedback (PREREGISTRATION-ICA.md).

Level 0 builds on the released battery of Kowalski et al. (arXiv 2608.21664,
github.com/mkobalski/activation-control, MIT): the model transcribes a fixed sentence,
so any change in the monitor readout is internal. The pure logic lives in
battery/prompts/alignment/monitor/stats. The model-facing parts sit behind the
`Backend` protocol, so everything except `hf_backend` runs on CPU with a fake backend.
"""
