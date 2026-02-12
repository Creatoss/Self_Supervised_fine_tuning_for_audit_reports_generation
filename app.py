
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import shutil

# --- Configuration ---
MODEL_DIR = "Models/Self_Supervised_finetuning_Model/audit-mistral-7b-qlora"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1" # Fallback base model
MAX_NEW_TOKENS = 512

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

def find_model_path(root_dir):
    """
    Recursively finds the first directory containing a model config file.
    """
    if not os.path.exists(root_dir):
        return None
    for root, dirs, files in os.walk(root_dir):
        if "adapter_config.json" in files or "config.json" in files:
            return root
    return None

def load_model():
    """
    Loads the model by searching for config files.
    """
    print(f"🔍 Searching for models in {MODEL_DIR}...")
    local_model_path = find_model_path(MODEL_DIR)

    if not local_model_path:
        print(f"⚠️ No local model found in {MODEL_DIR}. Please check your directory structure.")
        return None, None
        
    print(f"✅ Found model path at: {local_model_path}")
    
    # Check if it's an adapter (has adapter_config.json)
    is_adapter = os.path.exists(os.path.join(local_model_path, "adapter_config.json"))
    
    # Quantization only works on GPU (CUDA)
    # On CPU, bitsandbytes will crash or fail to load.
    bnb_config = None
    if device == "cuda":
        print("💡 GPU detected: Using 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("⚠️ CPU detected: Quantization disabled (not supported by bitsandbytes on CPU).")
        print("📦 Attempting to load in bfloat16 to save memory...")

    try:
        # Configuration for loading
        load_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if device == "cuda":
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            # On CPU, we use bfloat16 to fit in ~14GB
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = {"": "cpu"}

        if is_adapter:
            print(f"Loading Base Model ({device})...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                **load_kwargs
            )
            print("Loading Adapter...")
            model = PeftModel.from_pretrained(base_model, local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
            
        else:
            print(f"Loading Full Merged Model ({device})...")
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                **load_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
            
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Load model globally
model, tokenizer = load_model()

def generate_audit_report(instruction, context=""):
    """
    Generates text based on instruction and optional context.
    """
    if model is None:
        return "Error: Model not loaded. Check logs."
    
    prompt = f"### Instruction:\n{instruction}\n\n"
    if context:
        prompt += f"### Input:\n{context}\n\n"
    prompt += "### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part if possible
    if "### Response:\n" in response:
        response = response.split("### Response:\n")[1]
        
    return response.strip()

# --- Gradio UI ---
with gr.Blocks(title="Audit Report Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 AI Audit Report Generator")
    gr.Markdown("Generate professional audit report sections using a fine-tuned Mistral-7B model.")
    
    with gr.Row():
        with gr.Column(scale=1):
            instruction = gr.Textbox(
                label="Instruction / Query", 
                placeholder="e.g., Draft a section on Revenue Recognition...",
                lines=3
            )
            context = gr.Textbox(
                label="Context (Optional)",
                placeholder="Paste relevant financial data or facts here...",
                lines=5
            )
            generate_btn = gr.Button("🚀 Generate Report", variant="primary")
            
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Generated Report Section", 
                lines=15, 
                show_copy_button=True
            )
            
    generate_btn.click(
        fn=generate_audit_report, 
        inputs=[instruction, context], 
        outputs=output
    )
    
    gr.Markdown("### deployed via Hugging Face Spaces")

if __name__ == "__main__":
    demo.launch()
