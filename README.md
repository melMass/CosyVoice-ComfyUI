# CosyVoice-ComfyUI (mtb fork)
a comfyui custom node for [CosyVoice](https://github.com/FunAudioLLM/CosyVoice),you can find workflow in [workflows](./workflows/)

## new Feature
- suport `srt` file to single voice or mutiple voice clone
- a "dialogue" node, we the [dialogue example](./workflows/dialogue_workflow.json)

**input**
- [tts_srt](./workflows/dubbing/zh_test.srt)
- [prompt_wav](./workflows/dubbing/test.mp3)
- [zero_shot_prompt_wav](./workflows/zero_shot_prompt.wav)
- [cross_lingual_prompt_wav](./workflows/cross_lingual_prompt.wav)
- [prompt_srt](./workflows/dubbing/en_test.srt)(optional)

## Usage

```sh
# in ComfyUI/custom_nodes
git clone https://github.com/melMass/CosyVoice-ComfyUI.git
cd CosyVoice-ComfyUI
pip install -r requirements.txt
```

*weights will be downloaded from modelscope*
- [CosyVoice-300M](https://huggingface.co/model-scope/CosyVoice-300M)
- [CosyVoice-300M-Instruct](https://huggingface.co/model-scope/CosyVoice-300M-Instruct)
- [CosyVoice-300M-SFT](https://huggingface.co/model-scope/CosyVoice-300M-SFT)

## Preview
<video width=200 height=200 src="https://github.com/user-attachments/assets/25630131-2f4a-45bc-80e8-9b5acaf2f1c9"/>

## Thanks
- [CosyVoice-ComfyUI](https://github.com/AIFSH/CosyVoice-ComfyUI)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [CosyVoice_For_Windows](https://github.com/v3ucn/CosyVoice_For_Windows)
