# Creating a GitHub Hero GIF for Fractal-Mind

This guide walks through creating an engaging GitHub repository hero GIF that showcases the core value proposition: Fractal architecture for AI memory systems.

## GIF Goals

- **First impression**: Show code + architecture + performance
- **Technical audience**: Rust developers + ML engineers
- **Key message**: "Memory systems that scale like brains"
- **Length**: 8-12 seconds, loop seamlessly

---

## Recommended Tools

### Option 1: asciinema + FFmpeg (Best for code demos)

```bash
# Install asciinema
pip install asciinema

# Record terminal session
asciinema rec fractalmind_demo.cast

# Convert to GIF with custom dimensions
ffmpeg -f lavfi -i "color=c=black:s=600x300" -loop 0 fractalmind.gif
```

### Option 2: ttygif (Record existing terminal)

```bash
# Install
sudo apt install ttygif

# Record
ttygif -w 600 -h 300 -f 15

# Output: tty.gif
```

### Option 3: CLI tool - gips

```bash
# Install
cargo install gips

# Convert video to optimized GIF
gips --width 600 --height 300 --fps 20 input.mp4
```

---

## Scene Breakdown (8-12 seconds total)

### 0-2 seconds: Hero Shot (Code)
**Visual**: Rust code editor showing the key struct definitions

```rust
pub struct FractalNode {
    pub node_type: NodeType,      // Leaf | Parent | Root
    pub status: NodeStatus,       // Complete | Incomplete
    pub embedding: EmbeddingVec,  // 768D vector
    pub namespace: Namespace,     // global | user_<id>
}
```

**Text overlay**: "Fractal-Mind: AI Memory System in Rust"

### 2-4 seconds: Architecture Diagram
**Visual**: Simple animated flow showing the dual-phase architecture

```
┌─────────────┐     ┌─────────────────────┐
│   Vigilia   │ ──> │   Real-time Query   │
│ (Wakefulness)│     │   SSSP Navigation   │
└─────────────┘     └─────────────────────┘

┌─────────────┐     ┌─────────────────────┐
│     REM     │ ──> │  Async Learning     │
│   (Sleep)   │     │  Clustering         │
└─────────────┘     └─────────────────────┘
```

**Text overlay**: "Dual-Phase: Fast Queries + Autonomous Learning"

### 4-6 seconds: Performance Metrics
**Visual**: Terminal showing benchmark results

```bash
$ cargo bench

Running target/release/deps/benchmarks-xxxx
query_latency... 18ms (p95)
insert_throughput... 450/s
memory_footprint... 1.2GB (57% less than flat)
recall@10... 0.94
```

**Text overlay**: "57% lower memory, 57% faster queries"

### 6-8 seconds: Open Source CTA
**Visual**: GitHub repo page, star button animation

```
⭐ Star Fractal-Mind on GitHub
🐙 github.com/MadKoding/fractalmind
```

---

## Technical Specifications

- **Resolution**: 600x300 pixels (square-ish, fits GitHub hero)
- **Frame rate**: 15-20 FPS (smooth but file-size efficient)
- **Duration**: 8-12 seconds total
- **File size**: <500KB (GitHub's recommendation)
- **Loop**: Infinite loop (no fade out)

---

## FFmpeg Optimization Commands

```bash
# Best quality → smaller file size
ffmpeg -i input.mp4 -vf "fps=15,scale=600:300:flags=lanczos" -loop 0 -colors 256 fractalmind_optimized.gif

# Advanced: Optimize with palette
ffmpeg -i input.mp4 -vf "fps=15,scale=600:300:flags=lanczos,split[s0][s1];[s0]palettegen[pal];[s1][pal]paletteuse" -loop 0 fractalmind.gif

# Reduce file size further
gifsicle -i fractalmind.gif -O3 -o fractalmind_minimized.gif
```

---

## Quick Start Script

```bash
#!/bin/bash
# create_gif.sh - Generate optimized GitHub hero GIF

# 1. Record or prepare video (600x300 preferred)
#    - Use asciinema for terminal demos
#    - Use OBS for full-screen recordings

INPUT_VIDEO="${1:-fractalmind_demo.mp4}"
OUTPUT_GIF="fractalmind_hero.gif"

# 2. Convert to optimized GIF
ffmpeg -i "$INPUT_VIDEO" \
  -vf "fps=15,scale=600:300:flags=lanczos,split[s0][s1];[s0]palettegen[pal];[s1][pal]paletteuse" \
  -loop 0 \
  "$OUTPUT_GIF"

# 3. Verify size
ls -lh "$OUTPUT_GIF"

# 4. Optional: Minimize with gifsicle
if command -v gifsicle &> /dev/null; then
  gifsicle -i "$OUTPUT_GIF" -O3 -o "${OUTPUT_GIF%.gif}_min.gif"
  ls -lh "${OUTPUT_GIF%.gif}_min.gif"
fi

echo "✅ GIF created: $OUTPUT_GIF"
```

Usage:
```bash
chmod +x create_gif.sh
./create_gif.sh path/to/your/recording.mp4
```

---

## Alternative: Generate from Scratch with Python

If you don't have video footage, create a static GIF with text animations:

```python
from moviepy.editor import CaptionClip, CompositeVideoClip, ColorClip
import numpy as np

# Create background (black with subtle fractal pattern)
bg = ColorClip(size=(600, 300), color=[0, 0, 0], duration=10)

# Add text frames
code_clip = CaptionClip(
    text="pub struct FractalNode {...}",
    fontsize=24, font="DejaVu-Sans-Mono",
    color="white", bg_color="black",
    duration=2
).set_position(("center", "center"))

arch_clip = CaptionClip(
    text="Vigilia: SSSP Navigation → 18ms\nREM: Clustering → 450/s",
    fontsize=20, font="DejaVu-Sans-Mono",
    color="white", bg_color="black",
    duration=3
).set_position(("center", "center"))

# Combine and export as GIF
final = CompositeVideoClip([bg, code_clip, arch_clip])
final.write_gif("fractalmind_hero.gif", fps=15, program="ffmpeg")
```

---

## GitHub Upload

1. Save your GIF as `fractalmind_hero.gif`
2. GitHub automatically uses `hero.*` filenames in repo root
3. Or manually set via:
   - Repo → Settings → repository description
   - Or use GitHub API to update `organization/repo` settings

**Expected result**: When users visit your GitHub repo, the hero section shows your animated GIF instead of a static image.
