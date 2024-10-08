from collections.abc import Iterator
from typing import Literal, TypedDict
import torch
import random
import librosa
import torchaudio
import numpy as np
import numpy.typing as npt
import os
import sys
import comfy.utils
import folder_paths
import ffmpeg
import audiosegment
from srt import parse as srt_parse

from modelscope import snapshot_download
from pathlib import Path

now_dir = Path(__file__).parent.resolve()
sys.path.append(now_dir.as_posix())
sys.path.append((now_dir / "vendored").as_posix())

from cosyvoice.cli.cosyvoice import CosyVoice

input_dir = Path(folder_paths.get_input_directory())
output_dir = Path(folder_paths.get_output_directory()) / "cosyvoice_dubb"
pretrained_models = now_dir / "pretrained_models"

max_val = 0.8
prompt_sr, target_sr = 16000, 22050


ModelKind = Literal["base", "sft", "instruct", "25hz"]


class AudioTensor(TypedDict):
    sample_rate: int
    waveform: torch.Tensor


class CosyTTS(TypedDict):
    tts_speech: torch.Tensor


# This doesn't work since
# https://github.com/modelscope/modelscope/blob/51b33cecefed4daad3dccc47e0da60d5923ce8de/modelscope/hub/snapshot_download.py#L51
# MODEL_SETTINGS = {}
# with (now_dir / "models_settings.json").open(encoding="UTF-8") as source:
#     MODEL_SETTINGS = json.load(source)

model_prefix = "CosyVoice-300M"
MODEL_MAP = {
    "base": model_prefix,
    "sft": f"{model_prefix}-SFT",
    "instruct": f"{model_prefix}-Instruct",
    "25hz": f"{model_prefix}-25Hz",
}


def get_model(kind: ModelKind, *, auto_download: bool = True):
    # settings = MODEL_SETTINGS[kind]
    # name = settings["name"]
    # rev = settings["revision"]
    name = MODEL_MAP[kind]

    model_dir = pretrained_models / name
    if auto_download:
        snapshot_download(
            model_id=f"iic/{name}",
            local_dir=model_dir.as_posix(),
            # revision=rev,
        )
    if model_dir.exists():
        return model_dir.as_posix()


def STRING_INPUT(multi=False, default_val=""):
    if multi:
        return (
            "STRING",
            {"default": str(default_val), "multiline": True, "dynamicPrompts": True},
        )
    return ("STRING", {"default": str(default_val)})


DEFAULT_DIALOG = """
A: Hi I'm comfy, let's dance
B: No!
"""

sft_spk_list = [
    "Chinese Woman",
    "Chinese Man",
    "Japanese Man",
    "Cantonese Woman",
    "English Woman",
    "English Man",
    "Korean Woman",
]
sft_spk_list_cn = ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"]

sfts = dict(zip(sft_spk_list, sft_spk_list_cn))

inference_mode_list = [
    "Pre-trained tones",
    "3s Extreme Reproduction",
    "Cross-language reproduction",
    "Natural Language Control",
]
inference_mode_list_cn = ["预训练音色", "3s极速复刻", "跨语种复刻", "自然语言控制"]
# inference_modes = dict(zip(inference_mode_list, inference_mode_list_cn))


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess(speech: torch.Tensor, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def speed_change(input_audio: npt.NDArray[np.float32], speed: float, sr: int):
    # Check input data type and number of channels
    if input_audio.dtype != np.int16:
        raise ValueError("The input audio data type must be np.int16")
    if speed < 0:
        print(
            "Speed change error, a negative value was passed, returning original audio"
        )
        return input_audio

    # Convert to byte stream
    raw_audio = input_audio.astype(np.int16).tobytes()

    # Setting up the ffmpeg input stream
    input_stream = ffmpeg.input(
        "pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1
    )

    # Variable speed handling
    output_stream = input_stream.filter("atempo", speed)

    try:
        # Output stream to pipeline
        out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
            input=raw_audio, capture_stdout=True, capture_stderr=True
        )

        # Decode the pipeline output into NumPy arrays
        processed_audio = np.frombuffer(out, np.int16)

        return processed_audio
    except ffmpeg.Error as e:
        print("stdout:", e.stdout.decode("utf8"))
        print("stderr:", e.stderr.decode("utf8"))
        raise e


class TextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": STRING_INPUT(True)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "encode"

    CATEGORY = "CosyVoice"

    def encode(self, text):
        return (text,)


class CosyVoiceLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    ("base", "instruct", "sft", "25hz"),
                    {"default": "base"},
                ),
            },
            "optional": {"auto_download": ("BOOLEAN", {"default": True})},
        }

    RETURN_TYPES = ("COSYVOICE_MODEL",)
    RETURN_NAMES = ("model",)

    FUNCTION = "load"

    CATEGORY = "CosyVoice"

    def load(self, model: ModelKind, auto_download=True):
        model_dir = get_model(model, auto_download=auto_download)
        name = MODEL_MAP[model]
        if model_dir is None and auto_download is False:
            raise ValueError(f"""
            Model not found locally.
            You can:
            - set auto_download to True (this will download {name} to {pretrained_models})
            - move the files to {pretrained_models / name}

            To download the files manually check "https://www.modelscope.cn/models/iic/{name}"

            """)
        return (CosyVoice(model_dir),)


def ms(samples, sample_rate):
    return samples / sample_rate * 1000


class CosyBase:
    @classmethod
    def cut_audio(
        cls, audio: AudioTensor, max_length_ms=25000, *, report=False
    ) -> AudioTensor:
        sample_rate = audio["sample_rate"]
        input_duration = ms(len(audio["waveform"][0][0]), sample_rate)
        if report:
            print(f"Input duration: {input_duration} ms")

        max_length_samples = int(max_length_ms * sample_rate / 1000)
        cut_waveform = audio["waveform"][:, :, :max_length_samples]
        output_duration = ms(len(cut_waveform[0][0]), sample_rate)
        if report:
            print(f"Output duration: {output_duration} ms")
            reduction_percentage = (
                (input_duration - output_duration) / input_duration
            ) * 100
            print(f"Percentage of reduction: {reduction_percentage}%")

        return {
            "sample_rate": sample_rate,
            "waveform": cut_waveform,
        }

    @classmethod
    def to_comfy(cls, inference_output: Iterator[CosyTTS], speed: float = 1.0):
        tensors = []

        for inf in inference_output:
            output_tensor = inf["tts_speech"].squeeze(0)

            if speed != 1.0:
                output_numpy = (output_tensor * 32768).short().cpu().numpy()
                output_numpy = speed_change(output_numpy, speed, target_sr)
                output_tensor = torch.from_numpy(output_numpy).float() / 32768

            tensors.append(output_tensor)

        output = torch.cat(tensors, dim=-1).unsqueeze(0).unsqueeze(0)
        return {
            "waveform": output,
            "sample_rate": target_sr,
        }

    @classmethod
    def from_comfy(cls, audio: AudioTensor) -> torch.Tensor:
        waveform = audio["waveform"].squeeze(0)
        source_sr = audio["sample_rate"]
        speech = waveform.mean(dim=0, keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(
                orig_freq=source_sr, new_freq=prompt_sr
            )(speech)

        return speech


class CosyVoiceVc(CosyBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "source_speech": ("AUDIO",),
                "prompt_speech": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "process"

    CATEGORY = "CosyVoice"

    def process(
        self,
        model: CosyVoice,
        source_speech: AudioTensor,
        prompt_speech: AudioTensor,
        speed: float,
        seed: int,
    ):
        set_all_random_seed(seed)
        source = self.from_comfy(source_speech)
        prompt = self.from_comfy(prompt_speech)
        output = model.inference_vc(source, prompt, speed=speed)

        return (self.to_comfy(output),)


class CosyVoicePretrainedTones(CosyBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "tts_text": STRING_INPUT(True),
                "sft": (sft_spk_list, {"default": "English Woman"}),
                "speed": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "process"

    CATEGORY = "CosyVoice"

    def process(
        self, model: CosyVoice, tts_text: str, sft: str, speed: float, seed: int
    ):
        sft_cn = sfts[sft]
        set_all_random_seed(seed)
        output = model.inference_sft(tts_text, sft_cn, speed=speed)

        return (self.to_comfy(output),)


class CosyVoiceNaturalLanguageControl(CosyBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "instruct_text": STRING_INPUT(True),
                "tts_text": STRING_INPUT(True),
                "sft": (sft_spk_list, {"default": "English Woman"}),
                "speed": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    INPUT_NAMES = ("model (instruct)",)

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "process"

    CATEGORY = "CosyVoice"

    def process(
        self,
        model: CosyVoice,
        instruct_text: str,
        tts_text: str,
        sft: str,
        speed: float,
        seed: int,
    ):
        sft_cn = sfts[sft]
        set_all_random_seed(seed)
        output = model.inference_instruct(tts_text, sft_cn, instruct_text, speed=speed)
        audio = self.to_comfy(output)
        return (audio,)


class CosyVoiceCrossLanguageReproduction(CosyBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "prompt_wav": ("AUDIO",),
                "tts_text": STRING_INPUT(True),
                "speed": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    INPUT_NAMES = ("model",)

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "process"

    CATEGORY = "CosyVoice"

    def process(
        self,
        model: CosyVoice,
        prompt_wav: AudioTensor,
        tts_text: str,
        speed: float,
        seed: int,
    ):
        speech = self.from_comfy(prompt_wav)
        prompt_speech_16k = postprocess(speech)
        set_all_random_seed(seed)
        output = model.inference_cross_lingual(tts_text, prompt_speech_16k, speed=speed)
        audio = self.to_comfy(output)

        return (audio,)


class CosyVoice3SExtremeReproduction(CosyBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "prompt_text": STRING_INPUT(True),
                "prompt_wav": ("AUDIO",),
                "tts_text": STRING_INPUT(True),
                "speed": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    INPUT_NAMES = ("model",)

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)

    FUNCTION = "process"

    CATEGORY = "CosyVoice"

    def process(
        self,
        model: CosyVoice,
        prompt_text: str,
        prompt_wav: AudioTensor,
        tts_text: str,
        speed: float,
        seed: int,
    ):
        audio = self.cut_audio(prompt_wav, 3000, report=True)
        speech = self.from_comfy(audio)

        prompt_speech_16k = postprocess(speech)

        set_all_random_seed(seed)
        output = model.inference_zero_shot(
            tts_text, prompt_text, prompt_speech_16k, speed=speed
        )

        audio = self.to_comfy(output)

        return (audio,)


class CosyVoiceDialogue(CosyBase):
    """Generate dialogs from voice clones (dynamic inputs).

    Internal Preprocess:
        - Match sample rates to the highest.
        - Cut voices to 25s.
    """

    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "tts_text": STRING_INPUT(True, DEFAULT_DIALOG),
                "speed": ("FLOAT", {"default": 1.0}),
                "silence": (
                    "FLOAT",
                    {"default": 0.0, "min": -9999, "max": 9999, "step": 0.1},
                ),
                "seed": ("INT", {"default": 42}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    EXPERIMENTAL = True

    FUNCTION = "generate"

    CATEGORY = "CosyVoice"

    def map(self, voices):
        return {chr(65 + i): voice for i, voice in enumerate(voices)}

    def zip_dialog(self, tts_text, voice_map):
        lines = tts_text.split("\n")
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            speaker, dialog = line.split(":", 1)
            speaker = speaker.upper()
            if speaker in voice_map:
                voice_tensor = voice_map[speaker]
                result.append((speaker, voice_tensor, dialog))
        return result

    def sequence_audio(
        self, audios: list[AudioTensor], silence_duration: float
    ) -> AudioTensor:
        sample_rate = audios[0]["sample_rate"]

        for audio in audios:
            if audio["sample_rate"] != sample_rate:
                raise ValueError("All sample rates must be equal")

        sequence = []
        for i, audio in enumerate(audios):
            for waveform in audio["waveform"]:
                if i > 0 and silence_duration != 0:
                    if silence_duration > 0:
                        silence = torch.zeros((1, int(silence_duration * sample_rate)))
                        sequence.append(silence)
                    else:
                        overlap = int(abs(silence_duration) * sample_rate)
                        previous_audio = sequence[-1]

                        if (
                            previous_audio.shape[-1] < overlap
                            or waveform.shape[-1] < overlap
                        ):
                            actual_overlap = min(
                                previous_audio.shape[-1], waveform.shape[-1]
                            )
                            print(
                                f"Warning: Overlap duration exceeds audio segment length. Using actual overlap: {actual_overlap/sample_rate:.2f}s instead of {overlap/sample_rate:.2f}s."
                            )
                            overlap = actual_overlap

                        overlap_part = (
                            previous_audio[:, -overlap:] + waveform[:, :overlap]
                        )
                        if overlap_part.numel() == 0:
                            raise ValueError("Overlap part is empty or invalid.")

                        sequence[-1] = previous_audio[:, :-overlap]

                        waveform = waveform[:, overlap:]
                        sequence.append(overlap_part)

                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)

                sequence.append(waveform)

        sequenced_waveform = torch.cat(sequence, dim=-1).unsqueeze(0)

        return {"sample_rate": sample_rate, "waveform": sequenced_waveform}

    def generate(
        self,
        model: CosyVoice,
        tts_text: str,
        speed: float,
        silence: float,
        seed: int,
        **kwargs: AudioTensor,
    ) -> tuple[AudioTensor]:
        voices = [self.cut_audio(voice_tensor) for voice_tensor in kwargs.values()]
        voice_map = self.map(voices)
        text_map = self.zip_dialog(tts_text, voice_map)

        outputs = []

        pbar = comfy.utils.ProgressBar(len(text_map))
        for line in text_map:
            _speaker, voice_tensor, text = line

            speech = self.from_comfy(voice_tensor)
            prompt_speech_16k = postprocess(speech)

            set_all_random_seed(seed)
            infered = model.inference_cross_lingual(
                text, prompt_speech_16k, speed=speed
            )
            audio = self.to_comfy(infered)
            outputs.append(audio)
            pbar.update(1)

        stacked_voices = self.sequence_audio(outputs, silence)
        return (stacked_voices,)


class CosyVoiceNode(CosyBase):
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_text": STRING_INPUT(True),
                "speed": ("FLOAT", {"default": 1.0}),
                "inference_mode": (
                    inference_mode_list,
                    {"default": "Pre-trained tones"},
                ),
                "sft_dropdown": (sft_spk_list, {"default": "English Woman"}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "prompt_text": STRING_INPUT(),
                "prompt_wav": ("AUDIO",),
                "instruct_text": STRING_INPUT(),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("generated_audio",)
    DEPRECATED = True

    FUNCTION = "generate"

    # OUTPUT_NODE = False

    CATEGORY = "CosyVoice"

    def generate(
        self,
        tts_text,
        speed,
        inference_mode,
        sft_dropdown,
        seed,
        prompt_text=None,
        prompt_wav=None,
        instruct_text=None,
    ):
        sft_dropdown_cn = sfts[sft_dropdown]

        if inference_mode == "Natural Language Control":
            model_dir = pretrained_models / "CosyVoice-300M-Instruct"
            snapshot_download(
                model_id="iic/CosyVoice-300M-Instruct", local_dir=model_dir.as_posix()
            )
            if instruct_text is None:
                raise ValueError(
                    "in natural language control mode, instruct_text can't be none"
                )
        if inference_mode in ["Cross-language reproduction", "3s Extreme Reproduction"]:
            model_dir = pretrained_models / "CosyVoice-300M"
            snapshot_download(
                model_id="iic/CosyVoice-300M", local_dir=model_dir.as_posix()
            )

            if prompt_wav is None:
                raise ValueError(
                    "in 'Cross-language reproduction' or '3s Extreme Reproduction mode', prompt_wav can't be none"
                )
            if inference_mode == "3s Extreme Reproduction":
                if not prompt_text or len(prompt_text) == 0:
                    raise ValueError(
                        "The prompt text is empty, did you forget to enter the prompt text?"
                    )

        if inference_mode == "Pre-trained tones":
            model_dir = os.path.join(pretrained_models, "CosyVoice-300M-SFT")
            snapshot_download(
                model_id="iic/CosyVoice-300M-Instruct", local_dir=model_dir
            )

        if self.model_dir != model_dir:
            self.model_dir = model_dir
            self.cosyvoice = CosyVoice(model_dir)

        if prompt_wav:
            waveform = prompt_wav["waveform"].squeeze(0)
            source_sr = prompt_wav["sample_rate"]
            speech = waveform.mean(dim=0, keepdim=True)
            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(
                    orig_freq=source_sr, new_freq=prompt_sr
                )(speech)
        if inference_mode == "Pre-trained tones":
            print("get sft inference request")
            print(self.model_dir)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_sft(tts_text, sft_dropdown_cn)
        elif inference_mode == "3s Extreme Reproduction":
            print("get zero_shot inference request")
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_speech_16k
            )
        elif inference_mode == "Cross-language reproduction":
            print("get cross_lingual inference request")
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
        else:
            print("get instruct inference request")
            set_all_random_seed(seed)
            print(self.model_dir)
            output = self.cosyvoice.inference_instruct(
                tts_text, sft_dropdown_cn, instruct_text
            )

        audio = self.to_comfy(output, speed)
        return (audio,)


class CosyVoiceDubbingNode:
    def __init__(self):
        self.cosyvoice = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "tts_srt": ("SRT",),
                "prompt_wav": ("AUDIO",),
                "language": (
                    ["en", "fr", "ko", "jp", "yue", "zh"],
                    {"default": "en"},
                ),
                "if_single": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "prompt_srt": ("SRT",),
            },
        }

    RETURN_TYPES = ("AUDIO",)

    FUNCTION = "generate"

    CATEGORY = "CosyVoice"

    def generate(
        self,
        model: CosyVoice,
        tts_srt: str,
        prompt_wav: AudioTensor,
        language: str,
        if_single: bool,
        seed: int,
        prompt_srt: str | None = None,
    ):
        language = f"<|{language}|>"

        set_all_random_seed(seed)

        with open(tts_srt, "r", encoding="utf-8") as file:
            text_file_content = file.read()
        text_subtitles = list(srt_parse(text_file_content))

        if prompt_srt:
            with open(prompt_srt, "r", encoding="utf-8") as file:
                prompt_file_content = file.read()
            prompt_subtitles = list(srt_parse(prompt_file_content))

        waveform = prompt_wav["waveform"].squeeze(0)
        source_sr = prompt_wav["sample_rate"]
        speech = waveform.mean(dim=0, keepdim=True)

        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(
                orig_freq=source_sr, new_freq=prompt_sr
            )(speech)

        speech_numpy = speech.squeeze(0).numpy() * 32768
        speech_numpy = speech_numpy.astype(np.int16)
        audio_seg = audiosegment.from_numpy_array(speech_numpy, prompt_sr)

        if audio_seg.duration_seconds < 3:
            raise ValueError("prompt wav should be > 3s")

        # audio_seg.export(os.path.join(output_dir,"test.mp3"),format="mp3")

        pbar = comfy.utils.ProgressBar(len(text_subtitles))
        new_audio_seg = audiosegment.silent(0, target_sr)
        for i, text_sub in enumerate(text_subtitles):
            start_time = text_sub.start.total_seconds() * 1000
            end_time = text_sub.end.total_seconds() * 1000

            if i == 0:
                new_audio_seg += audio_seg[:start_time]

            if if_single:
                curr_tts_text = language + text_sub.content
            else:
                curr_tts_text = language + text_sub.content[1:]
                speaker_id = text_sub.content[0]

            prompt_wav_seg = audio_seg[start_time:end_time]
            if prompt_srt:
                prompt_text_list = [prompt_subtitles[i].content]
            while prompt_wav_seg.duration_seconds < 30:
                for j in range(i + 1, len(text_subtitles)):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (
                            audiosegment.silent(500, frame_rate=prompt_sr)
                            + audio_seg[j_start:j_end]
                        )
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (
                                audiosegment.silent(500, frame_rate=prompt_sr)
                                + audio_seg[j_start:j_end]
                            )
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)
                for j in range(0, i):
                    j_start = text_subtitles[j].start.total_seconds() * 1000
                    j_end = text_subtitles[j].end.total_seconds() * 1000
                    if if_single:
                        prompt_wav_seg += (
                            audiosegment.silent(500, frame_rate=prompt_sr)
                            + audio_seg[j_start:j_end]
                        )
                        if prompt_srt:
                            prompt_text_list.append(prompt_subtitles[j].content)
                    else:
                        if text_subtitles[j].content[0] == speaker_id:
                            prompt_wav_seg += (
                                audiosegment.silent(500, frame_rate=prompt_sr)
                                + audio_seg[j_start:j_end]
                            )
                            if prompt_srt:
                                prompt_text_list.append(prompt_subtitles[j].content)

                if prompt_wav_seg.duration_seconds > 3:
                    break
            print(f"prompt_wav {prompt_wav_seg.duration_seconds}s")

            output_dir.mkdir(exist_ok=True)
            prompt_wav_seg.export(
                (output_dir / f"{i}_prompt.wav").as_posix(), format="wav"
            )
            prompt_wav_seg_numpy = prompt_wav_seg.to_numpy_array() / 32768
            # print(prompt_wav_seg_numpy.shape)
            prompt_speech_16k = postprocess(
                torch.Tensor(prompt_wav_seg_numpy).unsqueeze(0)
            )
            if prompt_srt:
                # prompt_text = prompt_subtitles[i].content
                prompt_text = ",".join(prompt_text_list)
                print(f"prompt_text:{prompt_text}")
                curr_output = model.inference_zero_shot(
                    curr_tts_text, prompt_text, prompt_speech_16k
                )
            else:
                curr_output = model.inference_cross_lingual(
                    curr_tts_text, prompt_speech_16k
                )

            for out in curr_output:
                curr_output_numpy = out["tts_speech"].squeeze(0).numpy() * 32768
                curr_output_numpy = curr_output_numpy.astype(np.int16)
                text_audio = audiosegment.from_numpy_array(curr_output_numpy, target_sr)
                # text_audio.export(os.path.join(output_dir,f"{i}_res.wav"),format="wav")
                text_audio_dur_time = text_audio.duration_seconds * 1000

                if i < len(text_subtitles) - 1:
                    nxt_start = text_subtitles[i + 1].start.total_seconds() * 1000
                    dur_time = nxt_start - start_time
                else:
                    org_dur_time = audio_seg.duration_seconds * 1000
                    dur_time = org_dur_time - start_time

                ratio = text_audio_dur_time / dur_time

                if text_audio_dur_time > dur_time:
                    tmp_numpy = speed_change(curr_output_numpy, ratio, target_sr)
                    tmp_audio = audiosegment.from_numpy_array(tmp_numpy, target_sr)
                    # tmp_audio = self.map_vocal(text_audio,ratio,dur_time,f"{i}_res.wav")
                    tmp_audio += audiosegment.silent(
                        dur_time - tmp_audio.duration_seconds * 1000, target_sr
                    )
                else:
                    tmp_audio = text_audio + audiosegment.silent(
                        dur_time - text_audio_dur_time, target_sr
                    )

                new_audio_seg += tmp_audio

                if i == len(text_subtitles) - 1:
                    new_audio_seg += audio_seg[end_time:]
            pbar.update(1)

        output_numpy = new_audio_seg.to_numpy_array() / 32768
        # print(output_numpy.shape)
        audio = {
            "waveform": torch.stack([torch.Tensor(output_numpy).unsqueeze(0)]),
            "sample_rate": target_sr,
        }
        return (audio,)


class LoadSRT:
    @classmethod
    def INPUT_TYPES(cls):
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and f.split(".")[-1] in ["srt", "txt"]
        ]
        return {
            "required": {"srt": (sorted(files),)},
        }

    CATEGORY = "CosyVoice"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)
