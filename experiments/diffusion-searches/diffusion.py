"""
A minimal DDPM operating directly in the (normalized) 512-d font-ViT
embedding space, conditioned only on a precomputed text embedding and the
diffusion timestep. No visual information is ever fed to the model except
as the noised target itself -- this is deliberate: the point of the
experiment is to see how well text alone predicts P(visual embedding | text).
"""
import math

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """
    conditioning="concat" (original): cond is projected once and simply
    ADDED to the normalized hidden state -- conditioning enters at the
    same additive "location" as the residual signal itself, giving the
    network no structural push to actually use it over just modeling the
    marginal distribution of x. Measured consequence (see CLAUDE.md/
    experiments/diffusion-searches/README.md): text conditioning has a
    real but modest effect on the denoising loss (~1-7% relative), and
    the branching search built on top of this measured only ~47% top-1 /
    ~87% top-3 accuracy at the corpus tree's root.

    conditioning="film" (Perez et al. 2018, FiLM): cond instead predicts a
    per-channel scale and shift APPLIED to the normalized hidden state
    (h = norm(x) * (1 + gamma) + beta) before the block's own MLP -- the
    condition modulates the feature-processing pathway itself, rather
    than competing with it as another additive input. This was flagged as
    an unexplored, plausible fix in this project's own prior notes ("a
    weak concat-into-MLP; something like cross-attention or FiLM
    conditioning might use the text signal harder") and is the cheaper of
    the two to try. LayerNorm's own affine params are dropped, since
    FiLM's gamma/beta subsume that role.
    """

    def __init__(self, dim, condDim, conditioning="concat"):
        super().__init__()
        self.conditioning = conditioning
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        if conditioning == "film":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.filmProjection = nn.Linear(condDim, dim * 2)
        else:
            self.norm = nn.LayerNorm(dim)
            self.condProjection = nn.Linear(condDim, dim)

    def forward(self, x, cond):
        if self.conditioning == "film":
            gamma, beta = self.filmProjection(cond).chunk(2, dim=-1)
            h = self.norm(x) * (1.0 + gamma) + beta
        else:
            h = self.norm(x) + self.condProjection(cond)
        return x + self.net(h)


class DiffusionMLP(nn.Module):
    """
    x: noised visual embedding [B, visualDim]
    t: diffusion timestep [B]
    text: precomputed text embedding [B, textDim]
    """

    def __init__(self, visualDim=512, textDim=1024, hiddenDim=512, timeDim=128, depth=6, conditioning="concat"):
        super().__init__()
        self.conditioning = conditioning
        self.timeEmbed = nn.Sequential(
            SinusoidalTimestepEmbedding(timeDim),
            nn.Linear(timeDim, timeDim),
            nn.SiLU(),
            nn.Linear(timeDim, timeDim),
        )
        self.textProjection = nn.Sequential(
            nn.Linear(textDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, hiddenDim),
        )
        condDim = timeDim + hiddenDim

        self.inputProjection = nn.Linear(visualDim, hiddenDim)
        self.blocks = nn.ModuleList([ResidualBlock(hiddenDim, condDim, conditioning=conditioning)
                                      for _ in range(depth)])
        self.outputProjection = nn.Sequential(
            nn.LayerNorm(hiddenDim),
            nn.Linear(hiddenDim, visualDim),
        )

    def forward(self, x, t, text):
        tEmb = self.timeEmbed(t)
        textEmb = self.textProjection(text)
        cond = torch.cat([tEmb, textEmb], dim=-1)

        h = self.inputProjection(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.outputProjection(h)


class PlainResidualBlock(nn.Module):
    """
    Phase-1 block for TwoPhaseDiffusionMLP: pure residual processing of
    the content stream, no conditioning at all. Establishes a clean
    representation of the noised input before anything (timestep, text)
    touches it -- distinct from ResidualBlock, which mixes conditioning
    in at every single block from the first layer.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


class ConditionedResidualBlock(nn.Module):
    """
    Phase-2 block for TwoPhaseDiffusionMLP: conditioning injected by
    simple additive projection (added to the normalized hidden state),
    matching ResidualBlock's original "concat" mode -- deliberately NOT
    FiLM. FiLM was measured as a regression in the original architecture
    (diffusion_film checkpoint, CLAUDE.md) and there's no reason to expect
    it would help more in this smaller one.
    """

    def __init__(self, dim, condDim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.condProjection = nn.Linear(condDim, dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x, cond):
        h = self.norm(x) + self.condProjection(cond)
        return x + self.net(h)


class TwoPhaseDiffusionMLP(nn.Module):
    """
    "Shape I": a small number of PLAIN residual blocks (see
    PlainResidualBlock) process the noised visual input alone first --
    no conditioning at all -- establishing a clean representation of x
    before timestep/text ever touch it. A second group of blocks (see
    ConditionedResidualBlock) then injects conditioning via simple
    additive projection, not FiLM. Meant to be run much smaller than
    DiffusionMLP's default (hiddenDim=512, depth=6, ~4-5M params) --
    given the text conditioning signal itself has been repeatedly
    measured to be weak (near-zero-to-wrong-signed single-query text/
    visual correlation across five different sentence encoders), extra
    network capacity can only help fit whatever signal exists, it can't
    manufacture signal that isn't there.
    """

    def __init__(self, visualDim=64, textDim=1024, hiddenDim=128, timeDim=64,
                 numPlainBlocks=1, numConditionedBlocks=2):
        super().__init__()
        self.timeEmbed = nn.Sequential(
            SinusoidalTimestepEmbedding(timeDim),
            nn.Linear(timeDim, timeDim),
            nn.SiLU(),
            nn.Linear(timeDim, timeDim),
        )
        self.textProjection = nn.Sequential(
            nn.Linear(textDim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, hiddenDim),
        )
        condDim = timeDim + hiddenDim

        self.inputProjection = nn.Linear(visualDim, hiddenDim)
        self.plainBlocks = nn.ModuleList([PlainResidualBlock(hiddenDim) for _ in range(numPlainBlocks)])
        self.conditionedBlocks = nn.ModuleList([ConditionedResidualBlock(hiddenDim, condDim)
                                                  for _ in range(numConditionedBlocks)])
        self.outputProjection = nn.Sequential(
            nn.LayerNorm(hiddenDim),
            nn.Linear(hiddenDim, visualDim),
        )

    def forward(self, x, t, text):
        tEmb = self.timeEmbed(t)
        textEmb = self.textProjection(text)
        cond = torch.cat([tEmb, textEmb], dim=-1)

        h = self.inputProjection(x)
        for block in self.plainBlocks:
            h = block(h)
        for block in self.conditionedBlocks:
            h = block(h, cond)
        return self.outputProjection(h)


class GaussianDiffusion:
    """Standard DDPM schedule (Ho et al. 2020) with a linear beta schedule."""

    def __init__(self, timesteps=1000, betaStart=1e-4, betaEnd=2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(betaStart, betaEnd, timesteps, device=device)
        alphas = 1.0 - betas
        alphaBars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphaBars = alphaBars
        self.sqrtAlphaBars = torch.sqrt(alphaBars)
        self.sqrtOneMinusAlphaBars = torch.sqrt(1.0 - alphaBars)

    def to(self, device):
        for name in ["betas", "alphas", "alphaBars", "sqrtAlphaBars", "sqrtOneMinusAlphaBars"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def qSample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrtAlphaBar = self.sqrtAlphaBars[t].unsqueeze(-1)
        sqrtOneMinusAlphaBar = self.sqrtOneMinusAlphaBars[t].unsqueeze(-1)
        return sqrtAlphaBar * x0 + sqrtOneMinusAlphaBar * noise, noise

    def trainingLoss(self, model, x0, text, pairedText=None, consistencyWeight=0.0):
        """
        pairedText/consistencyWeight: an explicit same-font consistency
        regularizer (disabled by default, consistencyWeight=0.0). pairedText
        is a DIFFERENT query of the SAME font as `text` (see dataset.
        QueryFontDataset), noised with the exact same xt/noise/t draw so the
        two conditioned predictions are directly comparable. Standard
        training already gives an IMPLICIT version of this for free (every
        caption of a font shares the same regression target, so SGD already
        pushes all of them toward predicting consistent noise in
        expectation over the whole training run) -- this adds a DIRECT,
        per-batch penalty for the two captions' predictions disagreeing,
        rather than relying on that only emerging as a side effect of
        multi-task supervision. Targets the same diagnosed noise source as
        tag-presence conditioning (dataset.concatenateTagPresence): a
        single caption is a noisy, partial view of a font's true style,
        and different captions of the same font surface different subsets
        of it.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        xt, noise = self.qSample(x0, t)
        predicted = model(xt, t, text)
        loss = nn.functional.mse_loss(predicted, noise)
        if pairedText is not None and consistencyWeight > 0:
            predictedPaired = model(xt, t, pairedText)
            loss = loss + nn.functional.mse_loss(predictedPaired, noise)
            loss = loss + consistencyWeight * nn.functional.mse_loss(predicted, predictedPaired)
        return loss

    def respacedSteps(self, numSteps):
        """
        Evenly-spaced subset of the training schedule's timesteps, for
        sampling with fewer steps than training used (matching OpenAI's
        improved-diffusion "respacing" trick). The model is still queried at
        the ORIGINAL timestep values it was trained on -- only the reverse
        process's effective beta/alpha between consecutive selected steps is
        recomputed from the ratio of their alphaBars, so skipping steps stays
        a mathematically consistent shorter DDPM chain rather than an
        approximation that feeds the model out-of-distribution timesteps.
        """
        if numSteps >= self.timesteps:
            return list(range(self.timesteps))
        indices = torch.linspace(0, self.timesteps - 1, numSteps).round().long()
        return sorted(set(indices.tolist()))

    @torch.no_grad()
    def reverseSteps(self, model, text, visualDim, device=None, x=None, steps=None,
                      startIndex=None, stopIndex=0, guidanceScale=1.0, nullText=None):
        """
        Generator form of the reverse process over a respaced step schedule
        (see respacedSteps), yielding (scheduleIndex, x, x0Hat) after every
        update from startIndex down to stopIndex (inclusive). sample() is
        defined in terms of this for a single source of truth; the
        generator form exists so interactive callers -- experiments/
        diffusion-searches/branching.py, and eventually a web endpoint --
        can pause a trajectory at any schedule index, inspect/prune/
        resample the particle batch, and resume from exactly that point,
        rather than only ever running the process start-to-finish.

        x0Hat is the standard DDPM one-step estimate of the fully denoised
        result implied by this step's own noise prediction (no extra model
        call -- it reuses the same predictedNoise already computed for the
        ancestral update). It's blurrier than an actual completed sample
        (measured cosine ~0.8-0.94 against the true x_0 at a mid-schedule
        check, vs. real fonts' own ~0.96 nearest-neighbor tightness --
        different enough that it shouldn't be shown to a user as a final
        answer) but it's already at x_0's scale, unlike x itself, and it's
        free -- branching.py uses it to cheaply test which of a handful of
        FIXED, precomputed corpus regions (corpus.HierarchicalClusterIndex)
        the trajectory is currently heading toward, without needing a full
        rollout or any per-step clustering at all.

        On a full (non-respaced) schedule this reduces exactly to the
        standard DDPM reverse step: consecutive alphaBars' ratio is by
        definition that step's own alpha, so betaT below recovers the
        real per-step beta rather than an approximation.

        steps: the respaced schedule to use -- pass the SAME list across
        calls that resume one trajectory, since it fixes the per-step
        effective beta/alpha. Defaults to the full (unrespaced) schedule.
        x: starting state. Defaults to fresh Gaussian noise (x_T).
        startIndex: position in `steps` to start from. Defaults to the
        last index (i.e. x is x_T and every step is run).

        guidanceScale/nullText: classifier-free guidance (Ho & Salimans
        2022) -- predictedNoise = noiseUncond + guidanceScale * (noiseCond
        - noiseUncond), extrapolating each step's prediction away from a
        "pseudo-unconditional" prediction rather than using the
        conditioned prediction directly (guidanceScale=1, the default,
        disables this and reduces exactly to the original behavior). This
        model was never trained with real conditioning dropout, so
        nullText is a crude substitute (e.g. the corpus-wide mean text
        embedding) rather than a formally-learned unconditional
        distribution -- tested empirically, not assumed to be valid: a
        full-corpus coverage experiment (see experiments/diffusion-
        searches/README.md) found the bottleneck isn't reachability
        (~96%+ of the corpus is approximated within 2x real-neighbor
        tightness by SOME sample from SOME query already) but PRECISION
        (a font's own query lands near ITS OWN target only 7.45% of the
        time within 2x) -- exactly what CFG is meant to trade diversity
        for. Measured on 3000 held-out queries: guidanceScale 2-4 raised
        own-target %-within-2x from 7.9% to ~11.5-11.6% (a real, if
        modest, ~1.5x); scale 8+ overshoots and pushes samples off-
        manifold (realism check -- nearest-any-real-font distance --
        degrades sharply past scale ~6).
        """
        device = device or text.device
        if steps is None:
            steps = list(range(self.timesteps))
        if startIndex is None:
            startIndex = len(steps) - 1

        B = text.shape[0]
        if x is None:
            x = torch.randn(B, visualDim, device=device)

        stepsTensor = torch.tensor(steps, device=self.alphaBars.device)
        alphaBarAtStep = self.alphaBars[stepsTensor]
        prevAlphaBar = torch.cat([torch.ones(1, device=alphaBarAtStep.device), alphaBarAtStep[:-1]])

        for i in reversed(range(stopIndex, startIndex + 1)):
            t = torch.full((B,), steps[i], device=device, dtype=torch.long)
            alphaBarT = alphaBarAtStep[i]
            betaT = 1.0 - (alphaBarT / prevAlphaBar[i])
            alphaT = 1.0 - betaT

            predictedNoise = model(x, t, text)
            if guidanceScale != 1.0:
                noiseUncond = model(x, t, nullText)
                predictedNoise = noiseUncond + guidanceScale * (predictedNoise - noiseUncond)
            mean = (1.0 / torch.sqrt(alphaT)) * (
                x - (betaT / torch.sqrt(1.0 - alphaBarT)) * predictedNoise
            )
            x0Hat = (x - torch.sqrt(1.0 - alphaBarT) * predictedNoise) / torch.sqrt(alphaBarT)

            noise = torch.randn_like(x)
            mask = 1.0 if i > 0 else 0.0
            x = mean + mask * torch.sqrt(betaT) * noise

            yield i, x, x0Hat

    @torch.no_grad()
    def sample(self, model, text, visualDim, device=None, numSteps=None, guidanceScale=1.0, nullText=None):
        """
        Full reverse process, x_T ~ N(0, I) -> x_0. Pass numSteps < the
        schedule's training timesteps to sample with fewer steps than
        training used (e.g. to match a faster production configuration)
        without retraining -- see respacedSteps. See reverseSteps for
        guidanceScale/nullText (classifier-free guidance).
        """
        steps = self.respacedSteps(numSteps if numSteps is not None else self.timesteps)
        x = None
        for _, x, _ in self.reverseSteps(model, text, visualDim, device=device, steps=steps,
                                          guidanceScale=guidanceScale, nullText=nullText):
            pass
        return x
