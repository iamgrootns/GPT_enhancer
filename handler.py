import os
import torch
from transformers import pipeline
import runpod
import traceback


# --- Path for Initialization Error Logging ---
INIT_ERROR_FILE = "/tmp/init_error.log"

# --- Global Variables & Model Loading with Error Catching ---
pipe = None
lock = threading.Lock()  # ensures one job runs at a time

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading text-generation pipeline for openai/gpt-oss-20b...")

    # Load model with fp16 on GPU
    pipe = pipeline(
        "text-generation",
        model="openai/gpt-oss-20b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("✅ GPT-OSS model loaded successfully.")

except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize model: {tb_str}")
    pipe = None


# --- Core Logic ---
def enhance_prompt_with_gptoss(prompt: str, max_new_tokens: int = 300) -> str:
    instruction = (
        "You are a music prompt enhancer.\n\n"
        "Task:\n"
        "- Take the user’s raw text prompt about a vibe, scene, or style.\n"
        "- Rewrite it into a cinematic, descriptive, evocative music prompt.\n"
        "- Then structure the output into three clear sections with tags:\n"
        "  [Intro] → to set the mood and instrumentation,\n"
        "  [Interlude] → the evolving middle part,\n"
        "  [Outro] → a strong closing section.\n\n"
        "Rules:\n"
        "- Do NOT generate lyrics or vocals, only describe instruments, mood, pacing, energy.\n"
        "- Each section should feel distinct but connected.\n"
        "- Use vivid adjectives and music-related terms (tempo, intensity, instruments, ambience).\n"
        "- Keep the text concise and copy-paste ready for music generation.\n\n"
        "- Do Not Ever Give output of your Thinking , just enhance the prompt no need to write your thoughts in enhanced return"
        "Always end with:\n[Outro] <closing description>\n\n"
        "Output format:\n"
        "[Intro]\n...\n\n"
        "[Interlude]\n...\n\n"
        "[Outro]\n...\n\n"
        "User input:\n"
        f"{prompt}\n\n"
        "Enhanced structured version:\n"
    )

    # Generate output
    outputs = pipe(
        instruction,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )

    generated = outputs[0]["generated_text"].strip()

    # --- Post-processing ---
    unwanted_prefixes = [
        "Enhanced version:", "Output:", "Here's the enhanced version:",
        "Rewritten prompt:", "Enhanced prompt:", "\"We must", "We need to",
        "The enhanced version:", "Enhanced:"
    ]

    for prefix in unwanted_prefixes:
        if generated.startswith(prefix):
            generated = generated[len(prefix):].strip()

    lines = generated.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(phrase in line.lower() for phrase in
               ['we need to', 'must be', 'no repeating', 'so something like', 'but we need']):
            continue
        if line.startswith('"') and line.endswith('"'):
            cleaned_lines.append(line[1:-1])
        elif line.startswith('"'):
            cleaned_lines.append(line[1:])
        else:
            cleaned_lines.append(line)

    result = ""
    if cleaned_lines:
        result = max(cleaned_lines, key=len)
        if result.endswith((' speake', ' speak')):
            sentences = result.split('.')
            if len(sentences) > 1:
                result = '.'.join(sentences[:-1]) + '.'

    # Free GPU memory
    del outputs
    torch.cuda.empty_cache()

    return result if result else generated


# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            return {"error": f"Worker initialization failed: {f.read()}"}

    if pipe is None:
        return {"error": "Pipeline is not loaded."}

    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    max_new_tokens = job_input.get("max_new_tokens", 300)

    if not prompt:
        return {"error": "No 'prompt' provided in the input."}

    try:
        with lock:  # ensure only one job runs at a time
            enhanced_prompt = enhance_prompt_with_gptoss(prompt, max_new_tokens)
        return {"original": prompt, "enhanced": enhanced_prompt}

    except Exception:
        return {"error": f"An error occurred during enhancement: {traceback.format_exc()}"}


# --- Start Serverless Worker ---
runpod.serverless.start({"handler": handler})


