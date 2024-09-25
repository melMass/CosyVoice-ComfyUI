from .nodes import (
    TextNode,
    CosyVoiceNode,
    LoadSRT,
    CosyVoiceDubbingNode,
    CosyVoiceDialogue,
    CosyVoiceLoadModel,
    CosyVoiceNaturalLanguageControl,
    CosyVoice3SExtremeReproduction,
    CosyVoiceCrossLanguageReproduction,
    CosyVoicePretrainedTones,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT": LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode": CosyVoiceDubbingNode,
    "CosyVoiceDialog": CosyVoiceDialogue,
    "CosyVoiceLoadModel": CosyVoiceLoadModel,
    "CosyVoiceNaturalLanguageControl": CosyVoiceNaturalLanguageControl,
    "CosyVoice3SExtremeReproduction": CosyVoice3SExtremeReproduction,
    "CosyVoiceCrossLanguageReproduction": CosyVoiceCrossLanguageReproduction,
    "CosyVoicePretrainedTones": CosyVoicePretrainedTones,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceNode": "CosyVoiceNode [DEPRECATED]",
    "CosyVoiceNaturalLanguageControl": "CosyVoiceNaturalLanguageControl (Instruct)",
    "CosyVoice3SExtremeReproduction": "CosyVoice3SExtremeReproduction (Base)",
    "CosyVoiceCrossLanguageReproduction": "CosyVoiceCrossLanguageReproduction (Base)",
    "CosyVoiceDubbingNode": "CosyVoiceDubbingNode (Base)",
    "CosyVoicePretrainedTones": "CosyVoicePretrainedTones (SFT)",
    "CosyVoiceDialog": "CosyVoiceDialog (Base)",
}
