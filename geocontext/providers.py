"""Model adapter layer: different vendors, one interface.

Design principles (these affect experimental validity -- do not change casually):
1. **Every model receives exactly the same prompt text**, with no vendor-specific
   structured-output feature. Using one would conflate "model capability
   differences" with "vendor feature differences", and the comparison stops
   being clean.
2. Output is uniformly "let the model emit its own JSON, parse leniently";
   a parse failure is recorded as parse_error, never silently dropped.
3. No `thinking` parameter is set; each model runs under its own default
   behaviour -- this is the setting an ordinary user actually encounters.
   (Claude Opus 5 defaults to adaptive thinking on; Haiku 4.5 defaults off.
   The difference is recorded as-is in the paper.)
"""
import os
import base64
import mimetypes
from pathlib import Path

from . import prompts


def read_image(path):
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return p.read_bytes(), mime


# ---------------------------------------------------------------- providers

class Provider:
    """Subclasses implement _call(image_bytes, mime, prompt) -> (text, usage_dict)"""

    def __init__(self, model: str, name: str | None = None, max_tokens: int = 4000):
        self.model = model
        self.name = name or model
        self.max_tokens = max_tokens

    def query(self, image_path, lang="en", context="none", schema="v1"):
        """Evaluation entry point: sends the standard geolocation prompt."""
        return self.ask(image_path, prompts.build(lang, context, schema))

    def ask(self, image_path, prompt: str):
        """Send an arbitrary prompt. Used during benchmark construction (e.g.
        image-quality auditing), not during evaluation.

        Kept separate from `query` deliberately: an earlier version misused
        `query()` for image auditing, which sent the geolocation prompt instead
        of the audit prompt. The model then answered "where is this" while the
        parser was looking for a `has_overlay` field; a missing field parses as
        None, `bool(None)` is False, and all 80 images were recorded as "no
        overlay".
        """
        img, mime = read_image(image_path)
        text, usage = self._call(img, mime, prompt)
        return {"raw": text, "usage": usage}

    def _call(self, image_bytes, mime, prompt):
        raise NotImplementedError


class AnthropicProvider(Provider):
    """Claude. Usage follows the claude-api skill's python/claude-api/README.md.

    Use the full model-ID string with no date suffix: claude-opus-5 /
    claude-haiku-4-5. `effort` is sent only for models that support it (Haiku
    4.5 errors if it is sent).
    """

    EFFORT_OK = {"claude-opus-5", "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"}

    def __init__(self, model="claude-opus-5", effort=None, **kw):
        super().__init__(model, **kw)
        import anthropic  # lazy import, so its absence does not break other providers
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.effort = effort

    def _call(self, image_bytes, mime, prompt):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if self.effort and self.model in self.EFFORT_OK:
            kwargs["output_config"] = {"effort": self.effort}

        r = self.client.messages.create(**kwargs)
        # content is a list of content blocks, possibly including a thinking
        # block; keep only the text blocks.
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = {"input_tokens": r.usage.input_tokens,
                 "output_tokens": r.usage.output_tokens,
                 "stop_reason": r.stop_reason}
        return text, usage


def list_gemini_models():
    """List the Gemini models available under the current key.

    Model IDs get retired (gemini-2.5-flash is already closed to new users) --
    do not hard-code a guess. Run this before switching models to see what is
    actually available.
    """
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            out.append(m.name.replace("models/", ""))
    return sorted(out)


class GeminiProvider(Provider):
    """Gemini (google-genai SDK).

    Unlike the Anthropic section above, there is no authoritative reference
    doc for this one; it follows common google-genai usage. Before switching
    models, confirm the ID is still valid with list_gemini_models().
    """

    def __init__(self, model="gemini-3.6-flash", **kw):
        super().__init__(model, **kw)
        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call(self, image_bytes, mime, prompt):
        from google.genai import types
        r = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), prompt],
            # No tools are given, so automatic function calling has nothing to
            # do; leaving it enabled prints a warning on every call.
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        um = getattr(r, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "stop_reason": None,
        }
        return (r.text or ""), usage


class OpenRouterProvider(Provider):
    """Open-weight models go through OpenRouter (an OpenAI-compatible API)."""

    def __init__(self, model="qwen/qwen2.5-vl-72b-instruct", **kw):
        super().__init__(model, **kw)
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=os.environ["OPENROUTER_API_KEY"])

    def _call(self, image_bytes, mime, prompt):
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        )
        u = r.usage
        return r.choices[0].message.content or "", {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
            "stop_reason": r.choices[0].finish_reason,
        }


# Models to run are registered here. The short name on the left is written
# into the results file; do not rename casually, it invalidates the cache.
#
# NOTE: Gemini model IDs get retired -- gemini-2.5-flash is already closed to
# new users, with the API pointing to gemini-3.6-flash instead. Run
# list_gemini_models() before adding a new one.
REGISTRY = {
    # closed-weight frontier
    "claude-opus-5":    lambda: AnthropicProvider("claude-opus-5", effort="low"),
    "claude-haiku-4-5": lambda: AnthropicProvider("claude-haiku-4-5"),
    "gemini-flash":     lambda: GeminiProvider("gemini-3.6-flash"),
    # open-weight, one large and one small: ViewDiag found that smaller models
    # "collapse" more severely, so this lets us check whether brand/instance
    # confusion also scales with model size.
    "qwen3-vl-235b":    lambda: OpenRouterProvider("qwen/qwen3-vl-235b-a22b-instruct"),
    "qwen3-vl-8b":      lambda: OpenRouterProvider("qwen/qwen3-vl-8b-instruct"),
    # Text-only model, used only for benchmark construction (LLM auditing), not
    # evaluated. Text-only is deliberate: it can never be scored on this visual
    # task, which rules out "the auditor sets its own exam" from the ground up.
    "deepseek-v4-pro":  lambda: OpenRouterProvider("deepseek/deepseek-v4-pro"),
    # Non-reasoning auditor. deepseek-v4-pro is a **reasoning model**: auditing
    # a batch of 20 candidates burns 11,541 output tokens and takes 200s;
    # llama-3.3-70b does the same task in 643 tokens / 10s -- **20x faster,
    # 18x cheaper** -- with the same 20/20 parse success rate.
    # Meta was chosen over qwen-2.5-72b: the latter shares a vendor with the
    # evaluated qwen3-vl models, which would weaken the "the auditor must be
    # disjoint from the evaluated models" argument.
    "llama-3.3-70b":    lambda: OpenRouterProvider("meta-llama/llama-3.3-70b-instruct"),
    "deepseek-v4-flash": lambda: OpenRouterProvider("deepseek/deepseek-v4-flash"),
    # Vision model, used only for image-quality auditing, likewise not in the
    # evaluation set -- auditing images needs a VLM, but using one of the
    # evaluated VLMs to screen the test images would be circular.
    "deepseek-vision":  lambda: OpenRouterProvider("deepseek/deepseek-v4-flash-vision-exp"),
    #   ^ first choice, but requires enabling data sharing at
    #     openrouter.ai/settings/privacy, otherwise it 404s.
    # kimi is the fallback: measured to correctly read overlay text, and
    # Moonshot shares no vendor with any of the five evaluated models.
    # Do not use mistral-small -- measured to hallucinate a Taipei address on a
    # photo from the pilot site.
    "kimi-vision":      lambda: OpenRouterProvider("moonshotai/kimi-k2.5"),
}

#: The evaluated models. The auditor models must never appear in this list.
EVAL_MODELS = ["gemini-flash", "claude-haiku-4-5", "claude-opus-5",
               "qwen3-vl-235b", "qwen3-vl-8b"]
