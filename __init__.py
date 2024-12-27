from .nodes import (
    TextNode,
    CosyVoiceNode,
    LoadSRT,
    CosyVoiceDubbingNode,
    CosyVoiceDialogue,
    CosyVoiceDialogueV2,
    CosyVoiceVc,
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
    "CosyVoiceDialogV2": CosyVoiceDialogueV2,
    "CosyVoiceVc": CosyVoiceVc,
    "CosyVoiceLoadModel": CosyVoiceLoadModel,
    "CosyVoiceNaturalLanguageControl": CosyVoiceNaturalLanguageControl,
    "CosyVoice3SExtremeReproduction": CosyVoice3SExtremeReproduction,
    "CosyVoiceCrossLanguageReproduction": CosyVoiceCrossLanguageReproduction,
    "CosyVoicePretrainedTones": CosyVoicePretrainedTones,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CosyVoiceNode": "CosyVoiceNode [DEPRECATED]",
    "CosyVoiceNaturalLanguageControl": "CosyVoice NaturalLanguageControl (Instruct & 2.0 5b)",
    "CosyVoice3SExtremeReproduction": "CosyVoice3SExtremeReproduction (Base)",
    "CosyVoiceCrossLanguageReproduction": "CosyVoiceCrossLanguageReproduction (Base)",
    "CosyVoiceDubbingNode": "CosyVoiceDubbingNode (Base)",
    "CosyVoicePretrainedTones": "CosyVoice PretrainedTones (SFT)",
    "CosyVoiceDialog": "CosyVoice Dialog (Base)",
    "CosyVoiceDialogV2": "CosyVoice Dialog (V2)",
    "CosyVoiceVc": "CosyVoice VC (25Hz)",
}
