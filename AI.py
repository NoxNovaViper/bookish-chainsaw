# ==================== NeuroForge AI.py - Full Featured Version ====================
# neuroforge_asset.py — Complete Version with Iterations
import wave
import io
import os
import sys
import re
import json
import gc
import argparse
import shutil
import zipfile
import time
import base64
import struct
import zlib
import binascii
import math
from pathlib import Path
import hashlib

# PyTorch optional: if available we'll use it for tiny local models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional ;import F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------------------- Hardware / Safety Globals -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MAX_RAM_GB = 4
DEVICE = "cpu"
MAX_NEW_TOKENS_SAFE = 128

if TORCH_AVAILABLE:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

# ----------------------------- Byte Tokenizer -----------------------------
class ByteTokenizer:
    vocab_size = 256

    def encode(self, s: str):
        return list(s.encode('utf-8', errors='replace'))

    def decode(self, ids):
        try:
            return bytes(ids).decode('utf-8', errors='replace')
        except Exception:
            return "".join(chr(i % 256) for i in ids)

# ----------------------------- Ultra-small NanoTransformer (optional) -----------------------------
if TORCH_AVAILABLE:
    class NanoTransformer(nn.Module):
        def __init__(self, vocab_size=256, n_layer=1, n_head=1, n_embd=32, ctx_len=64):
            super().__init__()
            self.ctx_len = ctx_len
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Parameter(torch.zeros(1, ctx_len, n_embd))
            # Use TransformerEncoderLayer as a simple block
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=n_embd, nhead=n_head, dim_feedforward=n_embd*2,
                    dropout=0.0, activation='gelu', batch_first=True
                ) for _ in range(n_layer)
            ])
            self.ln = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx):
            B, T = idx.shape
            pos_emb = self.pos_emb[:, :T]
            x = self.tok_emb(idx) + pos_emb
            for block in self.blocks:
                x = block(x)
            x = self.ln(x)
            return self.head(x)
            
        def generate_improved(self, prompt, max_tokens=128, temperature=0.8, iteration=0):
            """Enhanced generation with iteration-based improvements"""
            idx = torch.tensor([self.encode(prompt)], device=self.device, dtype=torch.long)
            
            # Adjust creativity based on iteration
            temp = temperature * (1.0 + iteration * 0.1)
            
            with torch.no_grad():
                for _ in range(max_tokens):
                    idx_cond = idx[:, -self.ctx_len:]
                    logits = self(idx_cond)
                    logits = logits[:, -1, :] / temp
                    
                    # Better sampling with iteration improvements
                    if iteration > 2:
                        # Top-k sampling for later iterations
                        k = min(50, logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(logits, k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        next_idx = torch.multinomial(probs, num_samples=1)
                        next_id = top_k_indices.gather(-1, next_idx)
                    else:
                        # Standard sampling for early iterations
                        probs = F.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                    
                    idx = torch.cat((idx, next_id), dim=1)
                    
                    if idx.shape[1] > self.ctx_len * 2:
                        break
                        
            return self.decode(idx[0].cpu().tolist())

    def encode(self, s):
        return list(s.encode('utf-8', errors='replace'))
        
    def decode(self, ids):
        try:
            return bytes(ids).decode('utf-8', errors='replace')
        except Exception:
            return "".join(chr(i % 256) for i in ids)
else:
    NanoTransformer = None

# ----------------------------- NeuralAgent base -----------------------------
class NeuralAgent:
    def __init__(self, role, model_path=None):
        self.role = role
        self.model_path = model_path
        self.tk = ByteTokenizer()
        self.device = DEVICE
        self.model = None
        self.iteration_history = []

    def load_model(self):
        if not TORCH_AVAILABLE:
            self.model = None
            return
        if self.model is None:
            model = NanoTransformer(vocab_size=self.tk.vocab_size, n_layer=1, n_head=1, n_embd=32, ctx_len=64)
            model.to(self.device)
            if self.model_path and Path(self.model_path).exists():
                try:
                    model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                except Exception:
                    pass
            self.model = model.eval()

    def unload_model(self):
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            gc.collect()
            if TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def generate(self, prompt, max_new_tokens=None, iteration=0):
        """Enhanced generation with iteration support"""
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS_SAFE
            
        # Store iteration context
        self.iteration_history.append({"prompt": prompt[:100], "iteration": iteration})
        
        if TORCH_AVAILABLE and self.model is not None:
            # Use enhanced generation if available
            if hasattr(self.model, 'generate_improved'):
                out = self.model.generate_improved(prompt, max_new_tokens, iteration=iteration)
            else:
                with torch.no_grad():
                    idx = torch.tensor([self.tk.encode(prompt)], device=self.device, dtype=torch.long)
                    for _ in range(max_new_tokens):
                        idx_cond = idx[:, -self.model.ctx_len:]
                        logits = self.model(idx_cond)
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, next_id), dim=1)
                        if idx.shape[1] > self.model.ctx_len * 2:
                            break
                    out = self.tk.decode(idx[0].cpu().tolist())
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return out
        else:
            # Enhanced deterministic generation with iteration improvements
            out = f"// GENERATED by {self.role} (Iteration {iteration})\n// PROMPT: {prompt}\n"
            
            # Improve output based on iteration
            if iteration > 2:
                out += f"// ENHANCED VERSION (Iteration {iteration})\n"
            
            if "structure" in prompt.lower() or "files" in prompt.lower():
                # More sophisticated structure for later iterations
                modid = re.sub(r'\W+', '_', prompt.lower()).strip('_') or "neuro_mod"
                base_files = [
                    "build.gradle",
                    f"src/main/java/com/{modid}/{modid.capitalize()}Mod.java",
                    "src/main/resources/META-INF/mods.toml"
                ]
                
                if iteration >= 2:
                    # Add more files in later iterations
                    base_files.extend([
                        f"src/main/java/com/{modid}/item/{modid.capitalize()}Item.java",
                        f"src/main/java/com/{modid}/block/{modid.capitalize()}Block.java",
                        f"src/main/java/com/{modid}/entity/{modid.capitalize()}Entity.java",
                        f"src/main/resources/assets/{modid}/models/item/{modid}_item.json",
                        f"src/main/resources/assets/{modid}/lang/en_us.json",
                        f"src/main/resources/assets/{modid}/textures/item/{modid}_design.json",
                        f"src/main/resources/assets/{modid}/sounds/{modid}_design.json"
                    ])
                
                if iteration >= 4:
                    # Even more complexity for high iterations
                    base_files.extend([
                        f"src/main/java/com/{modid}/worldgen/{modid.capitalize()}Features.java",
                        f"src/main/java/com/{modid}/network/{modid.capitalize()}Network.java",
                        f"src/main/resources/data/{modid}/recipes/{modid}_recipes.json"
                    ])
                
                example = {"mod_id": modid, "files": base_files}
                out += json.dumps(example, indent=2)
            else:
                # Enhanced content generation
                base_content = (prompt + " ") * (2 + iteration)
                out += base_content
            
            return out[:max_new_tokens * (2 + iteration)]

# ----------------------------- Specialized Agents -----------------------------
class Architect(NeuralAgent):
    def think(self, prompt, iteration=0):
        structure_prompt = f"""// ROLE: Architecture Designer (Iteration {iteration})
// TASK: Create enhanced file structure for Minecraft Forge 1.20.1 mod: {prompt}
// ITERATION LEVEL: {iteration} (More complexity for higher iterations)
// Include comprehensive code files and asset placeholders.
// OUTPUT: Detailed JSON with mod_id and extensive files list.
"""
        raw = self.generate(structure_prompt, max_new_tokens=512 + (iteration * 128), iteration=iteration)
        
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                structure = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found")
        except Exception:
            # Fallback with iteration-based complexity
            modid = re.sub(r'\W+', '_', prompt.lower()).strip('_') or "neuro_mod"
            base_files = [
                "build.gradle",
                f"src/main/java/com/{modid}/{modid.capitalize()}Mod.java",
                "src/main/resources/META-INF/mods.toml"
            ]
            
            # Add complexity based on iteration
            if iteration >= 1:
                base_files.extend([
                    f"src/main/java/com/{modid}/item/{modid.capitalize()}Item.java",
                    f"src/main/java/com/{modid}/block/{modid.capitalize()}Block.java",
                    f"src/main/resources/assets/{modid}/models/item/{modid}_item.json",
                    f"src/main/resources/assets/{modid}/lang/en_us.json",
                    f"src/main/resources/assets/{modid}/textures/item/{modid}_design.json",
                    f"src/main/resources/assets/{modid}/sounds/{modid}_design.json"
                ])
            
            if iteration >= 2:
                base_files.extend([
                    f"src/main/java/com/{modid}/entity/{modid.capitalize()}Entity.java",
                    f"src/main/java/com/{modid}/gui/{modid.capitalize()}GUI.java",
                    f"src/main/resources/assets/{modid}/textures/gui/{modid}_gui_design.json",
                ])
            
            if iteration >= 3:
                base_files.extend([
                    f"src/main/java/com/{modid}/worldgen/{modid.capitalize()}Features.java",
                    f"src/main/java/com/{modid}/network/{modid.capitalize()}Network.java",
                    f"src/main/java/com/{modid}/event/{modid.capitalize()}Events.java",
                    f"src/main/resources/data/{modid}/recipes/{modid}_recipes.json",
                ])
            
            if iteration >= 4:
                base_files.extend([
                    f"src/main/java/com/{modid}/capability/{modid.capitalize()}Capabilities.java",
                    f"src/main/java/com/{modid}/client/{modid.capitalize()}Client.java",
                    f"src/main/java/com/{modid}/server/{modid.capitalize()}Server.java",
                    f"src/main/resources/assets/{modid}/textures/entity/{modid}_entity_design.json",
                    f"src/main/resources/assets/{modid}/sounds/ambient/{modid}_ambient_design.json",
                ])
            
            structure = {"mod_id": modid, "files": base_files, "iteration": iteration}
        
        return json.dumps(structure, indent=2)

class Coder(NeuralAgent):
    def think(self, file_path, context, iteration=0):
        enhanced_prompt = f"""// ROLE: Enhanced Coder (Iteration {iteration})
// FILE: {file_path}
// CONTEXT: {context[:500]}
// TASK: Generate high-quality code with iteration-based improvements
"""
        
        if file_path.endswith("build.gradle"):
            base_gradle = """plugins {
    id 'java'
    id 'net.minecraftforge.gradle' version '5.1.4'
}
group 'com.example'
version '1.0.0'
archivesBaseName = 'neuro-mod'

java.toolchain.languageVersion = JavaLanguageVersion.of(17)

dependencies {
    minecraft 'net.minecraftforge:forge:1.20.1-47.2.0'
}"""
            
            if iteration >= 2:
                base_gradle += """
    
// Enhanced dependencies for iteration """ + str(iteration) + """
repositories {
    maven { url = 'https://maven.theillusivec4.top/' }
}

dependencies {
    implementation fg.deobf('top.theillusivec4.curios:curios-forge:5.4.7+1.20.1')
}"""
            
            return base_gradle
            
        elif file_path.endswith(".java"):
            pkg = "com.example"
            clazz = Path(file_path).stem
            parts = Path(file_path).parts
            if len(parts) >= 3:
                modid = parts[-3] if parts[-3] != "java" else "example_mod"
            else:
                modid = "example_mod"
                
            base_java = f"""package {pkg};

import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.fml.ModLoadingContext;
import net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext;

@Mod("{modid}")
public class {clazz} {{
    public static final String MODID = "{modid}";
    
    public {clazz}() {{
        IEventBus modEventBus = FMLJavaModLoadingContext.get().getModEventBus();
        // Iteration {iteration} enhancements
"""
            
            if iteration >= 2:
                base_java += f"""        
        // Enhanced initialization for iteration {iteration}
        ModLoadingContext.get().registerConfig(net.minecraftforge.fml.config.ModConfig.Type.COMMON, Config.SPEC);
"""
            
            if iteration >= 3:
                base_java += f"""
        // Advanced features for iteration {iteration}
        modEventBus.addListener(this::setup);
        modEventBus.addListener(this::clientSetup);
"""
            
            base_java += """    }
"""
            
            if iteration >= 3:
                base_java += f"""
    private void setup(final net.minecraftforge.fml.event.lifecycle.FMLCommonSetupEvent event) {{
        // Common setup for iteration {iteration}
    }}
    
    private void clientSetup(final net.minecraftforge.fml.event.lifecycle.FMLClientSetupEvent event) {{
        // Client setup for iteration {iteration}
    }}
"""
            
            base_java += "}"
            return base_java
            
        elif file_path.endswith("mods.toml"):
            base_toml = """modLoader="javafml"
loaderVersion="[47,)"
license="MIT"

[[mods]]
modId="neuro_mod"
version="1.0.0"
displayName="NeuroForge Mod"
description="AI Generated Minecraft Mod"

[[dependencies.neuro_mod]]
modId="forge"
mandatory=true
versionRange="[47,)"
ordering="NONE"
side="BOTH"
"""
            
            if iteration >= 2:
                base_toml += f"""
# Enhanced metadata for iteration {iteration}
authors="NeuroForge AI"
credits="Generated by advanced AI system"
"""
            
            return base_toml
            
        elif file_path.endswith(".json") and "lang" in file_path:
            base_lang = {"item.neuro_mod_item": "Neural Item"}
            
            if iteration >= 2:
                base_lang.update({
                    "item.neuro_mod_sword": "Neural Sword",
                    "item.neuro_mod_bow": "Neural Bow",
                    "block.neuro_mod_block": "Neural Block"
                })
            
            if iteration >= 3:
                base_lang.update({
                    "entity.neuro_mod_entity": "Neural Entity",
                    "gui.neuro_mod_title": "Neural Interface"
                })
                
            return json.dumps(base_lang, indent=2)
            
        elif file_path.endswith("_design.json"):
            return json.dumps({
                "design": f"placeholder_iteration_{iteration}",
                "complexity": iteration,
                "enhanced": iteration >= 2
            }, indent=2)
        
        # Default enhanced generation
        return self.generate(enhanced_prompt, max_new_tokens=256 + (iteration * 128), iteration=iteration)

class Validator(NeuralAgent):
    def think(self, code, iteration=0):
        # Enhanced validation based on iteration
        issues = []
        
        if ".java" in str(code):
            if "@Mod" not in code:
                issues.append("Missing @Mod annotation")
            if "public class" not in code:
                issues.append("Missing public class declaration")
            if iteration >= 2 and "IEventBus" not in code:
                issues.append(f"Iteration {iteration}: Missing event bus setup")
            if iteration >= 3 and "setup" not in code:
                issues.append(f"Iteration {iteration}: Missing setup methods")
        
        if "build.gradle" in str(code):
            if "minecraft" not in code:
                issues.append("Missing minecraft dependency")
            if iteration >= 2 and "repositories" not in code:
                issues.append(f"Iteration {iteration}: Missing enhanced repositories")
        
        if issues:
            return "ERROR: " + "; ".join(issues)
        else:
            return f"OK (Iteration {iteration} validation passed)"

# ----------------------------- Asset Generation Helpers (Enhanced) -----------------------------
def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk = struct.pack("!I", len(data)) + chunk_type + data
    crc = struct.pack("!I", binascii.crc32(chunk_type + data) & 0xffffffff)
    return chunk + crc

def rgb_pixels_to_png_bytes(width: int, height: int, pixels: bytes) -> bytes:
    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += _png_chunk(b'IHDR', ihdr)
    row_bytes = []
    stride = width*3
    for y in range(height):
        start = y*stride
        row = b'\x00' + pixels[start:start+stride]
        row_bytes.append(row)
    raw = b''.join(row_bytes)
    comp = zlib.compress(raw, level=6)
    png += _png_chunk(b'IDAT', comp)
    png += _png_chunk(b'IEND', b'')
    return png

def generate_enhanced_texture_pixels(prompt: str, size: int = 16, iteration: int = 0, palette: list = None):
    """Enhanced texture generation with iteration-based improvements"""
    W, H = size, size
    default_palette = [
        (255, 0, 255),   # placeholder magenta
        (200, 200, 200), # blade/stone
        (60, 30, 10),    # hilt brown
        (255, 220, 0),   # yellow lightning
        (20, 120, 220),  # blue accent / crystal
        (0, 0, 0),       # black
        (255, 255, 255), # white
        (255, 100, 100), # red accent
        (100, 255, 100), # green accent
        (100, 100, 255), # blue accent
    ]
    
    if not palette:
        palette = default_palette
    
    # Enhanced palette for higher iterations
    if iteration >= 2:
        palette.extend([
            (180, 120, 60),   # bronze
            (255, 215, 0),    # gold
            (192, 192, 192),  # silver
            (128, 0, 128),    # purple
        ])
    
    if iteration >= 4:
        palette.extend([
            (255, 69, 0),     # orange red
            (50, 205, 50),    # lime green
            (30, 144, 255),   # dodger blue
            (255, 20, 147),   # deep pink
        ])

    canvas = [palette[0] for _ in range(W*H)]
    t = prompt.lower()

    # Enhanced pattern generation based on iteration
    complexity = min(iteration + 1, 5)
    
    # Base patterns
    if "sword" in t or "sao" in t or "aincrad" in t:
        # Create sword blade
        blade_width = 1 + (complexity // 2)
        for y in range(2, H-4):
            for bw in range(blade_width):
                if (W//2 - bw//2) < W:
                    canvas[y*W + (W//2 - bw//2)] = palette[1]
        
        # Create hilt
        hilt_start = max(0, H-4)
        for y in range(hilt_start, H):
            for x in range(max(0, W//2-3), min(W, W//2+4)):
                if 0 <= x < W:
                    canvas[y*W + x] = palette[2]
        
        # Add iteration-based enhancements
        if iteration >= 2:
            # Add gem in hilt
            gem_x, gem_y = W//2, H-2
            if 0 <= gem_x < W and 0 <= gem_y < H:
                canvas[gem_y*W + gem_x] = palette[4]
        
        if iteration >= 3:
            # Add blade pattern
            for y in range(3, H-5, 2):
                if 0 <= W//2+1 < W:
                    canvas[y*W + (W//2+1)] = palette[3]
        
        if iteration >= 4:
            # Add enhanced details
            for y in range(1, H-4):
                if y % 3 == 0 and W//2-1 >= 0:
                    canvas[y*W + (W//2-1)] = palette[7]  # red accent

    # Bow pattern with iterations
    if "bow" in t:
        # Basic bow structure
        for i in range(4, 12):
            if W//2 < W:
                canvas[i*W + (W//2)] = palette[1]  # handle
        
        # Bow string
        string_points = [(2,6), (3,7), (4,8), (5,9), (6,10), (7,11), (8,12), (9,11), (10,10), (11,9), (12,8), (13,7), (14,6)]
        for x, y in string_points:
            if 0 <= x < W and 0 <= y < H:
                canvas[y*W + x] = palette[5]  # string color
        
        if iteration >= 2:
            # Add bow arms
            for i in range(2, 6):
                left_x = W//2 - 3 + i
                right_x = W//2 + 3 - i
                if 0 <= left_x < W:
                    canvas[(i+4)*W + left_x] = palette[2]
                if 0 <= right_x < W:
                    canvas[(i+4)*W + right_x] = palette[2]

    # Crystal/magic patterns
    if "crystal" in t or "magic" in t:
        center_x, center_y = W//2, H//2
        crystal_size = 3 + (iteration // 2)
        
        for i in range(max(0, center_y-crystal_size), min(H, center_y+crystal_size+1)):
            for j in range(max(0, center_x-crystal_size), min(W, center_x+crystal_size+1)):
                distance = abs(i-center_y) + abs(j-center_x)
                if distance <= crystal_size:
                    canvas[i*W + j] = palette[4 + (distance % 3)]

    # Lightning/electrical effects
    if "lightning" in t or "electric" in t:
        lightning_coords = [(2,12),(3,10),(4,11),(5,9),(6,10),(7,8),(8,9),(9,7),(10,8),(11,6)]
        for (x,y) in lightning_coords:
            if 0 <= x < W and 0 <= y < H:
                canvas[y*W + x] = palette[3]
        
        if iteration >= 3:
            # Add secondary lightning
            for (x,y) in lightning_coords:
                if 0 <= x+1 < W and 0 <= y+1 < H:
                    canvas[(y+1)*W + (x+1)] = palette[7]

    # SAO/Anime specific enhancements
    if "sao" in t or "anime" in t or "aincrad" in t:
        # Add distinctive SAO color scheme
        sao_colors = {
            "interface": (0, 162, 255),    # SAO blue
            "warning": (255, 69, 0),       # SAO orange
            "health": (255, 20, 20),       # SAO red
            "mana": (20, 20, 255),         # SAO blue
        }
        
        # Replace some colors with SAO palette
        if iteration >= 1:
            palette[3] = sao_colors["interface"]
        if iteration >= 2:
            palette[7] = sao_colors["warning"]
        if iteration >= 3:
            palette[8] = sao_colors["health"]
            palette[9] = sao_colors["mana"]
        
        # Add SAO-style interface elements
        if iteration >= 4:
            # Add corner decorations
            corners = [(0,0), (W-1,0), (0,H-1), (W-1,H-1)]
            for x, y in corners:
                canvas[y*W + x] = palette[3]  # SAO blue

    # Floor-specific enhancements for Aincrad
    if "floor" in t or "aincrad" in t:
        # Add floor number indicator (simple pattern)
        if iteration >= 2:
            # Add a simple floor indicator pattern
            indicator_y = 1
            for x in range(1, min(6, W-1)):
                if x < W:
                    canvas[indicator_y*W + x] = palette[3]

    # Apply color themes based on keywords
    color_themes = {
        "fire": (255, 80, 0),
        "ice": (100, 200, 255), 
        "poison": (100, 255, 50),
        "dark": (80, 0, 120),
        "emerald": (20, 160, 80),
        "green": (20, 160, 80),
        "shadow": (50, 50, 50),
        "light": (255, 255, 200),
        "blood": (150, 20, 20),
        "arcane": (128, 0, 255),
    }
    
    for keyword, color in color_themes.items():
        if keyword in t:
            # Apply theme color to placeholder pixels
            canvas = [color if c == palette[0] else c for c in canvas]
            if iteration >= 2:
                # Add accent colors
                accent_color = tuple(min(255, c + 30) for c in color)
                canvas = [accent_color if i % (17 - iteration*2) == 0 and c == color else c 
                         for i, c in enumerate(canvas)]

    # Pack to bytes
    rgb = bytearray()
    for (r,g,b) in canvas:
        rgb.extend((r,g,b))

    return W, H, bytes(rgb)

def generate_enhanced_sound_wav_bytes(frequency=440.0, duration=0.5, volume=0.6, 
                                    sample_rate=22050, iteration=0, sound_type="default"):
    """Enhanced sound generation with iteration improvements"""
    # Enhance parameters based on iteration
    if iteration >= 2:
        duration += 0.2
        sample_rate = max(sample_rate, 44100)
    
    if iteration >= 3:
        volume = min(0.8, volume + 0.1)
    
    if iteration >= 4:
        # Add harmonics
        n_samples = int(duration * sample_rate)
        samples = []
        
        for i in range(n_samples):
            t = i / sample_rate
            
            # Base frequency
            base_wave = volume * math.sin(2*math.pi*frequency*t)
            
            # Add harmonics for complexity
            harmonic1 = (volume * 0.3) * math.sin(2*math.pi*frequency*2*t)
            harmonic2 = (volume * 0.2) * math.sin(2*math.pi*frequency*3*t)
            
            # Envelope for natural sound decay
            envelope = 1.0 - (t / duration) ** 2
            
            # Combine waves
            sample = (base_wave + harmonic1 + harmonic2) * envelope
            
            # Add sound type specific modifications
            if "sword" in sound_type.lower():
                # Sharp attack
                if t < 0.05:
                    sample *= (t / 0.05) ** 0.3
            elif "magic" in sound_type.lower():
                # Mystical vibrato
                vibrato = 1.0 + 0.1 * math.sin(2*math.pi*8*t)
                sample *= vibrato
            
            samples.append(sample)
        
        # Convert to WAV bytes
        with io.BytesIO() as tmpbuf:
            with wave.open(tmpbuf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                for sample in samples:
                    s = int(max(-1.0, min(1.0, sample)) * 32767)
                    wf.writeframes(struct.pack('<h', s))
            return tmpbuf.getvalue()
    
    else:
        # Simple generation for lower iterations
        n_samples = int(duration * sample_rate)
        with io.BytesIO() as tmpbuf:
            with wave.open(tmpbuf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                for i in range(n_samples):
                    t = i / sample_rate
                    val = volume * math.sin(2*math.pi*frequency*t) * (1.0 - t/duration)
                    s = int(max(-1.0, min(1.0, val)) * 32767)
                    wf.writeframes(struct.pack('<h', s))
            return tmpbuf.getvalue()

# ----------------------------- Enhanced Asset Designers -----------------------------
class TextureDesigner(NeuralAgent):
    def think(self, item_name, prompt, iteration=0):
        """Enhanced texture design with iteration improvements"""
        print(f"🎨 Designing texture for '{item_name}' (Iteration {iteration})")
        
        # Determine size based on iteration
        base_size = 16
        if iteration >= 3:
            base_size = 32
        elif iteration >= 5:
            base_size = 64
            
        W, H, pixels = generate_enhanced_texture_pixels(prompt, base_size, iteration)
        png = rgb_pixels_to_png_bytes(W, H, pixels)
        
        design = {
            "item_name": item_name,
            "iteration": iteration,
            "complexity_level": min(iteration + 1, 5),
            "dimensions": {"width": W, "height": H},
            "enhancements": [],
            "png_b64": base64.b64encode(png).decode('ascii')
        }
        
        # Document enhancements by iteration
        if iteration >= 1:
            design["enhancements"].append("Basic pattern generation")
        if iteration >= 2:
            design["enhancements"].append("Color theme application")
        if iteration >= 3:
            design["enhancements"].append("Detail overlays and accents")
        if iteration >= 4:
            design["enhancements"].append("SAO-specific styling")
        if iteration >= 5:
            design["enhancements"].append("Maximum detail and effects")
            
        design["note"] = f"Enhanced {W}x{H} texture for '{item_name}' (Iteration {iteration})"
        
        return json.dumps(design, indent=2)

class SoundDesigner(NeuralAgent):
    def think(self, sound_name, prompt, iteration=0):
        """Enhanced sound design with iteration improvements"""
        print(f"🔊 Designing sound for '{sound_name}' (Iteration {iteration})")
        
        # Determine sound properties based on prompt and iteration
        base_freq = 440
        if "sword" in prompt.lower():
            base_freq = 880  # Higher for metallic sounds
        elif "bow" in prompt.lower():
            base_freq = 660  # Medium for bow sounds
        elif "magic" in prompt.lower() or "crystal" in prompt.lower():
            base_freq = 1320  # High for magical sounds
        elif "lightning" in prompt.lower():
            base_freq = 1760  # Very high for electrical
        
        # Adjust frequency based on iteration
        freq_variation = base_freq + (iteration * 20)
        
        # Generate enhanced audio
        wav = generate_enhanced_sound_wav_bytes(
            frequency=freq_variation,
            duration=0.5 + (iteration * 0.1),
            volume=0.6,
            sample_rate=22050 + (iteration * 11025),
            iteration=iteration,
            sound_type=sound_name
        )
        
        design = {
            "sound_name": sound_name,
            "type": "enhanced_procedural",
            "iteration": iteration,
            "properties": {
                "frequency": freq_variation,
                "duration": 0.5 + (iteration * 0.1),
                "sample_rate": 22050 + (iteration * 11025),
                "complexity": min(iteration + 1, 5)
            },
            "enhancements": [],
            "wav_b64": base64.b64encode(wav).decode('ascii')
        }
        
        # Document audio enhancements
        if iteration >= 1:
            design["enhancements"].append("Basic waveform generation")
        if iteration >= 2:
            design["enhancements"].append("Extended duration and higher sample rate")
        if iteration >= 3:
            design["enhancements"].append("Volume optimization")
        if iteration >= 4:
            design["enhancements"].append("Harmonic complexity and sound typing")
        if iteration >= 5:
            design["enhancements"].append("Advanced envelope and effects")
            
        design["note"] = f"Enhanced audio for '{sound_name}' (Iteration {iteration})"
        
        return json.dumps(design, indent=2)

# ----------------------------- Enhanced Cloud Storage -----------------------------
class SimpleCloud:
    def __init__(self, base_dir="cloud_storage", use_s3=False, s3_bucket=None):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.upload_history = []
        
        if self.use_s3:
            try:
                import boto3
                self.s3 = boto3.client('s3')
            except Exception as e:
                print("⚠️ boto3 not available; falling back to local storage:", e)
                self.use_s3 = False

    def upload_bytes(self, key, data: bytes, iteration=0):
        upload_info = {
            "key": key,
            "size": len(data),
            "iteration": iteration,
            "timestamp": time.time()
        }
        
        if self.use_s3 and self.s3_bucket:
            try:
                self.s3.put_object(Bucket=self.s3_bucket, Key=key, Body=data)
                print(f"☁️ [Iter {iteration}] Uploaded to s3://{self.s3_bucket}/{key} ({len(data)} bytes)")
                upload_info["location"] = f"s3://{self.s3_bucket}/{key}"
                self.upload_history.append(upload_info)
                return upload_info["location"]
            except Exception as e:
                print("⚠️ S3 upload failed:", e)
        
        # Local fallback
        tgt = self.base / key
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_bytes(data)
        upload_info["location"] = str(tgt)
        self.upload_history.append(upload_info)
        print(f"☁️ [Iter {iteration}] Local storage: {tgt} ({len(data)} bytes)")
        return str(tgt)

    def get_iteration_summary(self):
        """Get summary of uploads by iteration"""
        summary = {}
        for upload in self.upload_history:
            iter_num = upload.get("iteration", 0)
            if iter_num not in summary:
                summary[iter_num] = {"count": 0, "total_size": 0, "files": []}
            summary[iter_num]["count"] += 1
            summary[iter_num]["total_size"] += upload["size"]
            summary[iter_num]["files"].append(upload["key"])
        return summary

# ----------------------------- Enhanced AssetGenerator -----------------------------
class AssetGenerator:
    def __init__(self, build_dir: Path, cloud: SimpleCloud = None, max_iterations=5):
        self.build_dir = Path(build_dir)
        self.cloud = cloud or SimpleCloud()
        self.max_iterations = max_iterations
        # Progressive complexity for different iterations
        self.texture_stages = {
            1: [16],
            2: [16, 32],
            3: [16, 32],
            4: [16, 32, 64],
            5: [16, 32, 64, 128]
        }
        self.sound_stages = {
            1: [22050],
            2: [22050, 44100],
            3: [22050, 44100],
            4: [22050, 44100, 48000],
            5: [22050, 44100, 48000, 96000]
        }

    def generate_texture_pipeline(self, modid: str, rel_path: str, prompt: str, iteration: int = 0):
        """Enhanced texture pipeline with iteration-based improvements"""
        rel_path = Path(rel_path)
        out_dir = Path(self.build_dir) / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get stages for this iteration level
        stages = self.texture_stages.get(min(iteration + 1, 5), [16])
        
        print(f"🎨 [Iter {iteration}] Generating texture pipeline for {rel_path}")
        print(f"    Stages: {stages}")
        
        last_png = None
        for stage_idx, size in enumerate(stages):
            print(f"    🔄 Processing stage {stage_idx + 1}/{len(stages)} (Size: {size}x{size})")
            
            W, H, pixels = generate_enhanced_texture_pixels(prompt, size, iteration)
            png = rgb_pixels_to_png_bytes(W, H, pixels)
            
            # Save stage version
            stage_name = rel_path.with_name(f"{rel_path.stem}_iter{iteration}_stage{stage_idx}.png")
            stage_path = Path(self.build_dir) / stage_name
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_path.write_bytes(png)
            
            # Upload to cloud with iteration info
            cloud_key = f"{modid}/textures/iter_{iteration}/{stage_path.name}"
            self.cloud.upload_bytes(cloud_key, png, iteration=iteration)
            
            print(f"    ✅ Stage {stage_idx + 1} saved: {stage_path}")
            last_png = png
            gc.collect()
        
        # Save final version
        final_path = Path(self.build_dir) / rel_path
        final_path.write_bytes(last_png)
        print(f"🎯 Final texture saved: {final_path}")
        
        return str(final_path)

    def generate_sound_pipeline(self, modid: str, rel_path: str, prompt: str, iteration: int = 0):
        """Enhanced sound pipeline with iteration-based improvements"""
        rel_path = Path(rel_path)
        out_dir = Path(self.build_dir) / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get stages for this iteration level
        stages = self.sound_stages.get(min(iteration + 1, 5), [22050])
        
        print(f"🔊 [Iter {iteration}] Generating sound pipeline for {rel_path}")
        print(f"    Sample rates: {stages}")
        
        last_wav = None
        for stage_idx, sample_rate in enumerate(stages):
            print(f"    🔄 Processing stage {stage_idx + 1}/{len(stages)} (Sample rate: {sample_rate}Hz)")
            
            # Determine base frequency from prompt
            base_freq = 440
            if "sword" in prompt.lower() or "blade" in prompt.lower():
                base_freq = 880
            elif "magic" in prompt.lower() or "crystal" in prompt.lower():
                base_freq = 1320
            elif "lightning" in prompt.lower():
                base_freq = 1760
            
            # Generate enhanced audio
            wav = generate_enhanced_sound_wav_bytes(
                frequency=base_freq + (stage_idx * 30),
                duration=0.5 + (iteration * 0.1),
                volume=0.6,
                sample_rate=sample_rate,
                iteration=iteration,
                sound_type=Path(rel_path).stem
            )
            
            # Save stage version
            stage_name = rel_path.with_name(f"{rel_path.stem}_iter{iteration}_stage{stage_idx}.wav")
            stage_path = Path(self.build_dir) / stage_name
            stage_path.parent.mkdir(parents=True, exist_ok=True)
            stage_path.write_bytes(wav)
            
            # Upload to cloud
            cloud_key = f"{modid}/sounds/iter_{iteration}/{stage_path.name}"
            self.cloud.upload_bytes(cloud_key, wav, iteration=iteration)
            
            print(f"    ✅ Stage {stage_idx + 1} saved: {stage_path}")
            last_wav = wav
            gc.collect()
        
        # Save final version
        final_path = Path(self.build_dir) / rel_path
        final_path.write_bytes(last_wav)
        print(f"🎯 Final sound saved: {final_path}")
        
        return str(final_path)

    def get_generation_summary(self, iteration: int):
        """Get summary of asset generation for this iteration"""
        return self.cloud.get_iteration_summary()

# ----------------------------- Enhanced Packager -----------------------------
class Packager:
    def think(self, files: dict, mod_name: str, build_dir=None, iteration: int = 0):
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', mod_name)
        safe_name = safe_name[:50]

        if build_dir is None:
            build_dir = Path(f"build_{safe_name}_iter_{iteration}")
            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
        else:
            build_dir = Path(build_dir)

        print(f"📦 [Iter {iteration}] Packaging {len(files)} files for mod '{mod_name}'")
        print(f"    Build directory: {build_dir}")
        
        file_stats = {"java": 0, "json": 0, "png": 0, "wav": 0, "other": 0}
        total_size = 0
        
        for path, content in files.items():
            full_path = build_dir / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, bytes):
                full_path.write_bytes(content)
                size = len(content)
            else:
                content_str = str(content)
                full_path.write_text(content_str, encoding='utf-8')
                size = len(content_str.encode('utf-8'))
            
            total_size += size
            
            # Update statistics
            ext = Path(path).suffix.lower()
            if ext == '.java':
                file_stats["java"] += 1
            elif ext == '.json':
                file_stats["json"] += 1
            elif ext == '.png':
                file_stats["png"] += 1
            elif ext == '.wav':
                file_stats["wav"] += 1
            else:
                file_stats["other"] += 1

        print(f"    📊 File statistics: {file_stats}")
        print(f"    💾 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")

        # Create JAR with iteration info
        jar_name = f"{safe_name}_iter_{iteration}.jar"
        jar_path = Path(jar_name)
        
        with zipfile.ZipFile(jar_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as jar:
            for file in build_dir.rglob('*'):
                if file.is_file():
                    jar.write(file, file.relative_to(build_dir))
        
        jar_size = jar_path.stat().st_size
        compression_ratio = (1 - jar_size / total_size) * 100 if total_size > 0 else 0
        
        print(f"    🗜️ JAR created: {jar_path} ({jar_size:,} bytes, {compression_ratio:.1f}% compressed)")

        return str(jar_path), str(build_dir)

# ----------------------------- Enhanced NeuralRouter & Brain -----------------------------
class NeuralRouter:
    def __init__(self, agents_config):
        self.agents = {}
        self.agent_usage = {}  # Track agent usage across iterations
        
        for name, (agent_class, role, path) in agents_config.items():
            try:
                agent = agent_class(role, path)
                self.agents[name] = agent
                self.agent_usage[name] = {"loads": 0, "generations": 0, "total_tokens": 0}
            except Exception as e:
                print(f"⚠️ Failed to create agent {name}: {e}")
                agent = NeuralAgent(role, path)
                self.agents[name] = agent
                self.agent_usage[name] = {"loads": 0, "generations": 0, "total_tokens": 0}

    def get_agent(self, agent_name):
        agent = self.agents.get(agent_name)
        if agent:
            if hasattr(agent, 'load_model'):
                agent.load_model()
            self.agent_usage[agent_name]["loads"] += 1
        return agent

    def release_agent(self, agent_name):
        agent = self.agents.get(agent_name)
        if agent and hasattr(agent, 'unload_model'):
            agent.unload_model()

    def get_usage_summary(self):
        """Get summary of agent usage across all iterations"""
        return self.agent_usage

class Brain:
    def __init__(self, router: NeuralRouter, tmp_build_root="neuro_builds"):
        self.router = router
        self.working_memory = {}
        self.tmp_build_root = Path(tmp_build_root)
        self.tmp_build_root.mkdir(parents=True, exist_ok=True)
        self.iteration_stats = {}

    def get_context_for_file(self, file_path, generated_files, iteration=0):
        context = f"Project structure: {json.dumps(self.working_memory.get('structure', {}), indent=2)}"
        context += f"\nIteration: {iteration}"
        
        if generated_files:
            # Include more context for higher iterations
            context_files = min(len(generated_files), 2 + iteration)
            for path, content in list(generated_files.items())[-context_files:]:
                preview_length = 200 + (iteration * 50)
                context += f"\n\n// {path}:\n{str(content)[:preview_length]}...\n"
        return context

    def refine_code(self, coder_agent, code, feedback, iteration=0):
        refine_prompt = f"""// ROLE: Advanced Code Refiner (Iteration {iteration})
// ORIGINAL CODE: {code[:1000 + (iteration * 200)]}
// VALIDATOR FEEDBACK: {feedback}
// ITERATION: {iteration} (Apply iteration-specific improvements)
// TASK: Fix errors and enhance code quality with Forge 1.20.1 compatibility
"""
        return coder_agent.generate(refine_prompt, max_new_tokens=512 + (iteration * 128), iteration=iteration)

    def create_mod(self, prompt, mod_name="NeuroMod", iterations=1):
        """Enhanced mod creation with configurable iterations"""
        total_start_time = time.time()
        iterations = max(1, min(iterations, 10))  # Cap at 10 iterations for safety
        
        print(f"\n🧠 NeuroForge AI System Starting")
        print(f"🎯 Target: '{mod_name}' with {iterations} iterations")
        print(f"💭 Prompt: {prompt[:100]}...")
        print("=" * 80)
        
        best_result = None
        best_score = -1
        
        for iteration in range(iterations):
            iter_start_time = time.time()
            print(f"\n🔄 ITERATION {iteration + 1}/{iterations}")
            print("=" * 40)
            
            # Phase 1: Architecture Design
            print(f"\n--- Phase 1: Architecture Design (Iteration {iteration + 1}) ---")
            architect = self.router.get_agent('architect')
            structure_raw = architect.think(prompt, iteration=iteration)
            self.router.release_agent('architect')

            try:
                json_match = re.search(r'\{.*\}', structure_raw, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON structure found from architect")
                structure = json.loads(json_match.group())
                self.working_memory['structure'] = structure
                mod_id = structure.get('mod_id', mod_name)
                print(f"🏛️ [Iter {iteration + 1}] Designed {len(structure['files'])} files for mod '{mod_id}'")
                
                if iteration > 0:
                    improvement = len(structure['files']) - len(self.iteration_stats.get(iteration - 1, {}).get('file_count', 0))
                    print(f"    📈 File count improvement: +{improvement} files from previous iteration")
                    
            except Exception as e:
                print(f"❌ [Iter {iteration + 1}] Architecture error: {e}")
                if iteration == 0:  # If first iteration fails completely
                    return None, None
                else:
                    print(f"⚠️ Using previous iteration's structure")
                    continue

            # Phase 2: Code Implementation
            print(f"\n--- Phase 2: Code Implementation (Iteration {iteration + 1}) ---")
            coder = self.router.get_agent('coder')
            validator = self.router.get_agent('validator')

            generated_files = {}
            file_list = structure['files']
            validation_passes = 0
            
            for i, file_path in enumerate(file_list):
                print(f"[{i+1}/{len(file_list)}] [Iter {iteration + 1}] Generating {file_path}")
                context = self.get_context_for_file(file_path, generated_files, iteration)
                code = coder.think(file_path, context, iteration=iteration)
                
                # Enhanced validation with more attempts for higher iterations
                max_attempts = 2 + (iteration // 2)
                for attempt in range(max_attempts):
                    feedback = validator.think(code, iteration=iteration)
                    if "ERROR" not in feedback:
                        validation_passes += 1
                        print(f"    ✅ Validation passed (Attempt {attempt + 1})")
                        break
                    else:
                        print(f"    ⚠️ Validation issue (Attempt {attempt + 1}): {feedback}")
                        if attempt < max_attempts - 1:
                            code = self.refine_code(coder, code, feedback, iteration)
                        else:
                            print(f"    ⚠️ Max validation attempts reached, proceeding...")
                
                generated_files[file_path] = code

            self.router.release_agent('coder')
            self.router.release_agent('validator')
            
            validation_rate = (validation_passes / len(file_list)) * 100
            print(f"📊 [Iter {iteration + 1}] Validation success rate: {validation_rate:.1f}%")

            # Setup build directory for this iteration
            build_dir = self.tmp_build_root / f"build_{mod_id}_iter_{iteration + 1}"
            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            
            # Write generated files
            for path, content in generated_files.items():
                full_path = build_dir / path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    full_path.write_bytes(content)
                else:
                    full_path.write_text(content, encoding='utf-8')

            # Phase 3: Enhanced Asset Generation
            print(f"\n--- Phase 3: Asset Generation (Iteration {iteration + 1}) ---")
            cloud = SimpleCloud(base_dir=str(build_dir / "cloud_storage"))
            asset_gen = AssetGenerator(build_dir, cloud=cloud, max_iterations=iterations)

            texture_count = 0
            sound_count = 0
            
            # Process design placeholders
            for path in list(structure['files']):
                p = Path(path)
                if p.suffix == ".json" and ("textures" in path and "_design" in path):
                    rel_png = str(Path(path).with_name(Path(path).stem.replace("_design","") + ".png"))
                    name = Path(path).stem
                    prompt_txt = f"{mod_id} {name} {prompt}"
                    print(f"🎨 [Iter {iteration + 1}] Generating texture: {path} -> {rel_png}")
                    asset_gen.generate_texture_pipeline(mod_id, rel_png, prompt_txt, iteration=iteration)
                    
                    if (build_dir / rel_png).exists():
                        png_bytes = (build_dir / rel_png).read_bytes()
                        generated_files[rel_png] = png_bytes
                        texture_count += 1
                        
                elif p.suffix == ".json" and ("sounds" in path and "_design" in path):
                    rel_wav = str(Path(path).with_name(Path(path).stem.replace("_design","") + ".wav"))
                    name = Path(path).stem
                    prompt_txt = f"{mod_id} {name} {prompt}"
                    print(f"🔊 [Iter {iteration + 1}] Generating sound: {path} -> {rel_wav}")
                    asset_gen.generate_sound_pipeline(mod_id, rel_wav, prompt_txt, iteration=iteration)
                    
                    if (build_dir / rel_wav).exists():
                        wav_bytes = (build_dir / rel_wav).read_bytes()
                        generated_files[rel_wav] = wav_bytes
                        sound_count += 1

            print(f"🎨 [Iter {iteration + 1}] Generated {texture_count} textures, {sound_count} sounds")

            # Phase 4: Packaging
            print(f"\n--- Phase 4: Packaging (Iteration {iteration + 1}) ---")
            packager = Packager()
            jar_path, final_build_dir = packager.think(generated_files, mod_id, build_dir, iteration=iteration)
            
            # Calculate iteration score (simple heuristic)
            iter_score = (len(generated_files) * 0.3) + (validation_rate * 0.4) + (texture_count * 0.2) + (sound_count * 0.1)
            
            # Store iteration statistics
            iter_time = time.time() - iter_start_time
            self.iteration_stats[iteration] = {
                "file_count": len(generated_files),
                "validation_rate": validation_rate,
                "texture_count": texture_count,
                "sound_count": sound_count,
                "score": iter_score,
                "time": iter_time,
                "jar_path": jar_path,
                "build_dir": final_build_dir
            }
            
            print(f"📊 [Iter {iteration + 1}] Score: {iter_score:.2f}, Time: {iter_time:.1f}s")
            
            # Track best result
            if iter_score > best_score:
                best_score = iter_score
                best_result = (jar_path, final_build_dir, iteration + 1)
                print(f"🏆 [Iter {iteration + 1}] New best result! (Score: {iter_score:.2f})")
            
            # Memory cleanup between iterations
            gc.collect()
            if TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # Final summary
        total_time = time.time() - total_start_time
        print("\n" + "=" * 80)
        print("🎉 NEUROFORGE GENERATION COMPLETE")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"🔄 Iterations completed: {iterations}")
        print(f"🏆 Best result: Iteration {best_result[2]} (Score: {best_score:.2f})")
        print(f"📦 Best JAR: {best_result[0]}")
        print(f"📁 Best source: {best_result[1]}")
        
        # Print iteration comparison
        print(f"\n📊 ITERATION COMPARISON:")
        for i in range(iterations):
            stats = self.iteration_stats[i]
            print(f"  Iter {i+1}: Score {stats['score']:.2f} | Files {stats['file_count']} | "
                  f"Textures {stats['texture_count']} | Sounds {stats['sound_count']} | "
                  f"Validation {stats['validation_rate']:.1f}%")
        
        return best_result[0], best_result[1]

# ----------------------------- Enhanced NeuroSymbolicDebugger -----------------------------
class NeuroSymbolicDebugger:
    def __init__(self, router, build_dir, crash_log_path, iterations=1):
        self.router = router
        self.build_dir = Path(build_dir)
        self.crash_log_path = Path(crash_log_path)
        self.iterations = iterations
        self.solutions = []
        self.debug_history = []

    def analyze_crash_log(self, iteration=0):
        """Enhanced crash log analysis with iteration-based improvements"""
        print(f"🔍 [Iter {iteration}] Analyzing crash log: {self.crash_log_path}")
        
        try:
            crash_log = self.crash_log_path.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return f"❌ Failed to read crash log: {e}"

        # Enhanced pattern matching with iteration improvements
        patterns = [
            (r'Caused by: (.*?)\n', "root_cause"),
            (r'at (.*?)\((.*?\.java:\d+)\)', "stack_trace"),
            (r'Exception in thread ".*?" (.*?)\n', "thread_exception"),
            (r'java\.lang\.(.*?):', "exception_type"),
        ]

        # Add more patterns for higher iterations
        if iteration >= 2:
            patterns.extend([
                (r'Could not execute entrypoint stage.*?due to errors!', "entrypoint_failure"),
                (r'Missing required dependency: (.*?) @', "missing_dependency"),
                (r'java\.lang\.NoSuchMethodError: (.*?)\n', "no_such_method"),
            ])
        
        if iteration >= 3:
            patterns.extend([
                (r'java\.lang\.ClassNotFoundException: (.*?)\n', "class_not_found"),
                (r'java\.lang\.AbstractMethodError: (.*?)\n', "abstract_method_error"),
                (r'java\.lang\.IncompatibleClassChangeError: (.*?)\n', "incompatible_class"),
            ])

        analysis = {"iteration": iteration, "patterns_found": []}
        for pattern, pattern_name in patterns:
            matches = re.findall(pattern, crash_log)
            if matches:
                analysis["patterns_found"].append({
                    "pattern": pattern_name,
                    "matches": matches[:3],  # Show first 3 matches
                    "count": len(matches)
                })

        # Enhanced solution generation
        debug_agent = self.router.get_agent('validator')
        prompt = f"""// ROLE: Advanced Debugger (Iteration {iteration})
// CRASH LOG ANALYSIS:
{json.dumps(analysis, indent=2)}
// TASK: Propose solutions to fix these errors
// ENHANCEMENTS: Apply iteration-specific improvements
"""
        solution = debug_agent.generate(prompt, max_new_tokens=512, iteration=iteration)
        self.router.release_agent('validator')
        
        self.solutions.append({
            "iteration": iteration,
            "analysis": analysis,
            "solution": solution
        })
        return solution

    def apply_solution(self, solution_index=0):
        """Apply a solution from the debug history"""
        if solution_index >= len(self.solutions):
            return "❌ Invalid solution index"
        
        solution = self.solutions[solution_index]
        print(f"🛠️ Applying solution from iteration {solution['iteration']}")
        return solution['solution']

    def get_best_solution(self):
        """Get the most recent solution"""
        if not self.solutions:
            return "No solutions available"
        return self.solutions[-1]

# ----------------------------- CLI / main ------------------------------------
def make_agent_config(models_dir="models"):
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    # create tiny placeholder model files only for a couple agents to save disk/RAM
    if TORCH_AVAILABLE:
        for name in ["architect", "coder"]:
            p = models_dir / f"{name}.pt"
            if not p.exists():
                dummy = NanoTransformer(n_layer=1, n_head=1, n_embd=32, ctx_len=64)
                torch.save(dummy.state_dict(), p)
    AGENT_CONFIG = {
        'architect': (Architect, "Designer", str(models_dir / "architect.pt")),
        'coder': (Coder, "Programmer", str(models_dir / "coder.pt")),
        'validator': (Validator, "Quality", None),
        # texture / sound designers use text-only generation and don't need heavy models
        'texture_designer': (TextureDesigner, "TextureDesigner", None),
        'sound_designer': (SoundDesigner, "SoundDesigner", None)
    }
    return AGENT_CONFIG

def main():
    parser = argparse.ArgumentParser(description="NeuroForge: Brain-inspired Minecraft Forge Mod Maker (with assets)")
    parser.add_argument("--generate", action="store_true", help="Generate a new mod")
    parser.add_argument("--debug", action="store_true", help="Debug a crashed mod")
    parser.add_argument("--prompt", type=str, help="Mod description for generation")
    parser.add_argument("--mod_name", type=str, default="NeuroMod", help="Name for generated mod")
    parser.add_argument("--iterations", type=int, default=3, help="Number of refinement iterations (1-10)")
    parser.add_argument("--crash", type=str, help="Path to crash log for debugging")
    parser.add_argument("--build_dir", type=str, help="Build directory for debugging")
    args = parser.parse_args()

    AGENT_CONFIG = make_agent_config()
    router = NeuralRouter(AGENT_CONFIG)
    cortex = Brain(router)

    if args.generate:
        if not args.prompt:
            print("❌ Missing --prompt for generation")
            return
        jar_path, build_dir = cortex.create_mod(args.prompt, args.mod_name, iterations=args.iterations)
        if jar_path:
            print(f"🎉 Done! Jar: {jar_path}")
        else:
            print("❌ Mod creation failed")
    elif args.debug:
        if not args.crash or not args.build_dir:
            print("❌ Missing --crash or --build_dir for debugging")
            return
        debugger = NeuroSymbolicDebugger(router, args.build_dir, args.crash, args.iterations)
        for i in range(args.iterations):
            print(f"\n🔄 Debug Iteration {i+1}/{args.iterations}")
            solution = debugger.analyze_crash_log(iteration=i)
            print(f"🧠 Proposed solution:\n{solution}")
        best_solution = debugger.get_best_solution()
        print(f"\n🔧 Applying best solution from iteration {best_solution['iteration']}")
        print(best_solution['solution'])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()