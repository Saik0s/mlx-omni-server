import tempfile
import uuid
from pathlib import Path

from f5_tts_mlx.generate import generate
from mlx_audio.tts.generate import generate_audio
from pydantic import BaseModel, Field
from typing_extensions import override

from .schema import TTSRequest


class TTSModelAdapter(BaseModel):
    """Base class to adapt different TTS models to support the audio endpoint."""

    path_or_hf_repo: str | Path = Field(
        None, title="The path or the huggingface repository to load the model from."
    )

    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        pass

    @classmethod
    def from_path_or_hf_repo(cls, path_or_hf_repo: str) -> "TTSModelAdapter":
        pid = (path_or_hf_repo or "").lower()
        if path_or_hf_repo == "lucasnewman/f5-tts-mlx":
            return F5Model(path_or_hf_repo=path_or_hf_repo)
        if "qwen3-tts" in pid:
            return Qwen3TTSModel(path_or_hf_repo=path_or_hf_repo)
        return MlxAudioModel(path_or_hf_repo=path_or_hf_repo)


class F5Model(TTSModelAdapter):

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        generate(
            model_name=request.model,
            generation_text=request.input,
            speed=request.speed,
            output_path=str(output_path),
            **(request.get_extra_params() or {}),
        )
        return Path(output_path).exists()


class MlxAudioModel(TTSModelAdapter):
    """Adapter for Kokoro-style models where `lang_code` is the first char of `voice`."""

    path_or_hf_repo: str = Field("mlx-community/Kokoro-82M-4bit")

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        voice = request.voice if hasattr(request, "voice") else "af_sky"
        lang_code = voice[:1]

        extra_params = request.get_extra_params() or {}

        generate_audio(
            text=request.input,
            model=self.path_or_hf_repo,
            voice=voice,
            speed=request.speed,
            lang_code=lang_code,
            file_prefix=str(output_path).rsplit(".", 1)[0],
            audio_format=request.response_format.value,
            sample_rate=24000,
            join_audio=True,
            verbose=False,
            **extra_params,
        )

        return Path(output_path).exists()


class Qwen3TTSModel(TTSModelAdapter):
    """Adapter for Alibaba Qwen3-TTS variants (Base / CustomVoice / VoiceDesign).

    Qwen3-TTS's `model.generate()` auto-routes based on `tts_model_type` in
    config.json. Client passes `voice` as speaker (e.g. 'Vivian', 'Ryan'),
    `instruct` via `extra_body` for emotion (CustomVoice) or voice description
    (VoiceDesign), and optional `lang_code` ('auto' default, or a language name
    like 'English', 'Russian', 'Chinese').
    """

    @override
    def generate_audio(self, request: TTSRequest, output_path: str | Path) -> bool:
        self.path_or_hf_repo = request.model
        voice = request.voice if (hasattr(request, "voice") and request.voice) else None

        extra_params = request.get_extra_params() or {}
        lang_code = extra_params.pop("lang_code", "auto")
        instruct = extra_params.pop("instruct", None)

        generate_audio(
            text=request.input,
            model=self.path_or_hf_repo,
            voice=voice,
            speed=request.speed,
            lang_code=lang_code,
            instruct=instruct,
            file_prefix=str(output_path).rsplit(".", 1)[0],
            audio_format=request.response_format.value,
            sample_rate=24000,
            join_audio=True,
            verbose=False,
            **extra_params,
        )

        return Path(output_path).exists()


class TTSService:
    model: TTSModelAdapter

    def __init__(self, path_or_hf_repo: str | Path | None = None):
        self.model = TTSModelAdapter.from_path_or_hf_repo(path_or_hf_repo)

    async def generate_speech(self, request: TTSRequest) -> bytes:
        # Re-resolve adapter per request so model id changes between calls work.
        self.model = TTSModelAdapter.from_path_or_hf_repo(request.model)

        ext = request.response_format.value
        tmp_dir = Path(tempfile.gettempdir())
        out_path = tmp_dir / f"mlx-omni-tts-{uuid.uuid4().hex}.{ext}"

        try:
            self.model.generate_audio(request=request, output_path=out_path)
        except Exception as e:
            raise Exception(f"TTS generation failed: {type(e).__name__}: {e}") from e

        # mlx-audio may append `_000` to file_prefix; find the actual file.
        actual = out_path if out_path.exists() else next(
            tmp_dir.glob(f"{out_path.stem}*.{ext}"), None
        )
        if actual is None or not actual.exists():
            raise Exception(
                f"TTS produced no output file near {out_path}. "
                "Check server logs for the underlying model error "
                "(common causes: unsupported voice for model, missing "
                "language code, or an mlx-audio model-loading failure "
                "that printed instead of raising)."
            )

        try:
            data = actual.read_bytes()
        finally:
            actual.unlink(missing_ok=True)
        return data
