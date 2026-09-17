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
    def __init__(self, dim, condDim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.condProjection = nn.Linear(condDim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, cond):
        h = self.norm(x) + self.condProjection(cond)
        return x + self.net(h)


class DiffusionMLP(nn.Module):
    """
    x: noised visual embedding [B, visualDim]
    t: diffusion timestep [B]
    text: precomputed text embedding [B, textDim]
    """

    def __init__(self, visualDim=512, textDim=1024, hiddenDim=512, timeDim=128, depth=6):
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
        self.blocks = nn.ModuleList([ResidualBlock(hiddenDim, condDim) for _ in range(depth)])
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

    def trainingLoss(self, model, x0, text):
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        xt, noise = self.qSample(x0, t)
        predicted = model(xt, t, text)
        return nn.functional.mse_loss(predicted, noise)

    @torch.no_grad()
    def pSample(self, model, xt, t, text):
        """One reverse step: p(x_{t-1} | x_t)."""
        betaT = self.betas[t].unsqueeze(-1)
        alphaT = self.alphas[t].unsqueeze(-1)
        alphaBarT = self.alphaBars[t].unsqueeze(-1)

        predictedNoise = model(xt, t, text)
        mean = (1.0 / torch.sqrt(alphaT)) * (
            xt - (betaT / torch.sqrt(1.0 - alphaBarT)) * predictedNoise
        )

        noise = torch.randn_like(xt)
        mask = (t > 0).float().unsqueeze(-1)
        return mean + mask * torch.sqrt(betaT) * noise

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
    def sample(self, model, text, visualDim, device=None, numSteps=None):
        """
        Full reverse process, x_T ~ N(0, I) -> x_0. Pass numSteps < the
        schedule's training timesteps to sample with fewer steps than
        training used (e.g. to match a faster production configuration)
        without retraining -- see respacedSteps.
        """
        device = device or text.device
        B = text.shape[0]
        x = torch.randn(B, visualDim, device=device)

        if numSteps is None or numSteps >= self.timesteps:
            for step in reversed(range(self.timesteps)):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                x = self.pSample(model, x, t, text)
            return x

        steps = self.respacedSteps(numSteps)
        alphaBarAtStep = self.alphaBars[torch.tensor(steps, device=self.alphaBars.device)]
        prevAlphaBar = torch.cat([torch.ones(1, device=alphaBarAtStep.device), alphaBarAtStep[:-1]])

        for i in reversed(range(len(steps))):
            t = torch.full((B,), steps[i], device=device, dtype=torch.long)
            alphaBarT = alphaBarAtStep[i]
            betaT = 1.0 - (alphaBarT / prevAlphaBar[i])
            alphaT = 1.0 - betaT

            predictedNoise = model(x, t, text)
            mean = (1.0 / torch.sqrt(alphaT)) * (
                x - (betaT / torch.sqrt(1.0 - alphaBarT)) * predictedNoise
            )

            noise = torch.randn_like(x)
            mask = 1.0 if i > 0 else 0.0
            x = mean + mask * torch.sqrt(betaT) * noise

        return x
