# %cd /content/omnivoice-colab
import os
import sys
import logging
import re
import uuid
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import soundfile as sf

temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# Path setup for Whisper auto-transcribe fallback (if present)
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
try:
    from subtitle import subtitle_maker
except ImportError:
    subtitle_maker = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging & Model Loading
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logging.getLogger("omnivoice").setLevel(logging.WARNING)

print("Loading model from k2-fsa/OmniVoice to cuda ...")
from hf_mirror import download_model

try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )
except Exception:
    omnivoice_model_path = download_model(
        "k2-fsa/OmniVoice",
        download_folder="./OmniVoice_Model",
        redownload=False,
        workers=6,
        use_snapshot=False,
    )
    model = OmniVoice.from_pretrained(
        omnivoice_model_path,
        device_map="cuda",
        dtype=torch.float16,
        load_asr=False,
    )

sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Multi-Format Subtitle Parser (.srt, .vtt, .ass)
# ---------------------------------------------------------------------------
def _parse_time_str(t_str):
    """Converts HH:MM:SS.mmm or MM:SS.mmm or H:MM:SS.cc to seconds."""
    t_str = t_str.strip().replace(',', '.')
    parts = t_str.split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def parse_subtitles(file_path):
    """Parses .srt, .vtt, and .ass files into list of (start_sec, end_sec, text)."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    entries = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.ass':
        for line in lines:
            if line.startswith("Dialogue:"):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    start_sec = _parse_time_str(parts[1])
                    end_sec = _parse_time_str(parts[2])
                    text = parts[9]
                    text = re.sub(r'\{.*?\}', '', text)  # remove ASS override tags
                    text = text.replace(r'\N', ' ').replace(r'\n', ' ').strip()
                    if text:
                        entries.append((start_sec, end_sec, text))

    else:  # Handles .srt and .vtt
        content = "".join(lines)
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            b_lines = [l.strip() for l in block.split('\n') if l.strip()]
            for i, line in enumerate(b_lines):
                if '-->' in line:
                    times = line.split('-->')
                    start_t = _parse_time_str(times[0].strip())
                    # VTT might have alignment metadata after timestamp
                    end_raw = times[1].strip().split(' ')[0]
                    end_t = _parse_time_str(end_raw)
                    text = " ".join(b_lines[i+1:])
                    text = re.sub(r'<[^>]+>', '', text).strip()  # remove HTML/VTT tags
                    if text:
                        entries.append((start_t, end_t, text))
                    break

    entries.sort(key=lambda x: x[0])
    return entries

# ---------------------------------------------------------------------------
# TTS Core Function
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

def run_dubbing(sub_file, ref_audio, ref_text, language, match_timing, progress=gr.Progress()):
    if not sub_file:
        return None, "Please upload a subtitle file (.srt, .vtt, .ass)", None
    if not ref_audio:
        return None, "Please upload reference audio for voice cloning.", None

    # Auto-transcribe reference text if not provided
    if not ref_text and subtitle_maker:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            res = subtitle_maker(ref_audio, whisper_lang)
            if res and len(res) > 7:
                ref_text = res[7]
        except Exception:
            pass

    entries = parse_subtitles(sub_file.name)
    if not entries:
        return None, "No dialogues found in the uploaded subtitle file.", None

    # Pre-calculate voice clone prompt once to save compute
    voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    lang = language if (language and language != "Auto") else None

    # Create empty audio canvas
    total_duration = entries[-1][1] + 1.5
    total_samples = int(total_duration * sampling_rate)
    timeline_audio = np.zeros(total_samples, dtype=np.float32)

    gen_config = OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    for idx, (start_sec, end_sec, text) in enumerate(progress.tqdm(entries, desc="Generating Dialogues")):
        dur = (end_sec - start_sec) if match_timing else None
        kw = {
            "text": text,
            "language": lang,
            "voice_clone_prompt": voice_prompt,
            "generation_config": gen_config
        }
        if dur and dur > 0:
            kw["duration"] = float(dur)

        try:
            audio = model.generate(**kw)
            waveform = audio[0].cpu().numpy().flatten()

            start_sample = int(start_sec * sampling_rate)
            end_sample = start_sample + len(waveform)

            if end_sample > len(timeline_audio):
                timeline_audio = np.pad(timeline_audio, (0, end_sample - len(timeline_audio)))

            timeline_audio[start_sample:end_sample] += waveform
        except Exception as e:
            print(f"Error on line {idx + 1}: {e}")

    # Normalize audio to prevent distortion
    max_val = np.max(np.abs(timeline_audio))
    if max_val > 1.0:
        timeline_audio = timeline_audio / max_val

    final_waveform = (timeline_audio * 32767).astype(np.int16)
    
    out_mp3 = f"{temp_audio_dir}/Dubbed_Output_{uuid.uuid4().hex[:6]}.mp3"
    sf.write(out_mp3, final_waveform, sampling_rate, format='MP3')

    return (sampling_rate, final_waveform), f"Complete! Synced {len(entries)} dialogue lines.", out_mp3

# ---------------------------------------------------------------------------
# Clean & Dedicated UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Subtitle TTS Mixer") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 15px auto;">
            <h2 style="font-size: 1.8em; margin-bottom: 2px;">🎙️ Subtitle Direct Audio Dubber</h2>
            <p style="color: #666;">Upload SRT, VTT, or ASS files along with reference voice to generate full mixed audio.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            sub_file_input = gr.File(
                label="Upload Subtitle File (.srt, .vtt, .ass)", 
                file_types=[".srt", ".vtt", ".ass"]
            )
            ref_audio_input = gr.Audio(
                label="Reference Audio (Voice to Clone)", 
                type="filepath"
            )
            ref_text_input = gr.Textbox(
                label="Reference Audio Text (Optional)", 
                placeholder="Leave blank for automatic detection"
            )
            lang_input = gr.Dropdown(
                label="Audio Language", 
                choices=_ALL_LANGUAGES, 
                value="Auto"
            )
            match_timing_input = gr.Checkbox(
                label="Strict Subtitle Time Match", 
                value=True, 
                info="Automatically speeds up or slows down lines to match subtitle timestamps."
            )
            generate_btn = gr.Button("Generate Full Mixed Audio", variant="primary")

        with gr.Column(scale=1):
            audio_preview = gr.Audio(label="Full Synced Audio Preview")
            status_box = gr.Textbox(label="Status", lines=2)
            mp3_download = gr.File(label="Download Final MP3")

    generate_btn.click(
        fn=run_dubbing,
        inputs=[sub_file_input, ref_audio_input, ref_text_input, lang_input, match_timing_input],
        outputs=[audio_preview, status_box, mp3_download]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
