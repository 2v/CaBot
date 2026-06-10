from configparser import ConfigParser
from pathlib import Path
from openai import OpenAI
import math
import json
import sys
import re
import os
import subprocess
import tempfile
import shutil
import multiprocessing
import time
import argparse
from typing import Dict, List, Tuple
from PIL import Image
import concurrent.futures
import base64

try:
    from .openai_retry import call_with_retry
except ImportError:          # also runnable as a plain script (see __main__ below)
    from openai_retry import call_with_retry

#sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, os.getcwd())

# Function to encode the image
def encode_image(image_path):
    # Open and encode the JPEG image (no TIF conversion needed)
    print(f"[DEBUG] encode_image called with path: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def resolve_base_path(base_path):
    """
    Robustly resolve a base path that could be relative or absolute.
    
    Args:
        base_path: Path string that could be relative (e.g., "../datasets/nejm_cpcs") 
                  or absolute (e.g., "/full/path/to/datasets")
    
    Returns:
        Path: Resolved absolute path
    """
    # Get project root: cabot_video.py is in cabot/, so parent.parent gets us to NEJMBench/
    # Structure: NEJMBench/cabot/cabot_video.py → NEJMBench/
    project_root = Path(__file__).parent.parent.absolute()
    
    base_path_obj = Path(base_path)
    
    if base_path_obj.is_absolute():
        return base_path_obj
    else:
        # For relative paths, resolve them relative to the project root
        # This works for any relative path format (../, ./, relative, etc.)
        return (project_root / base_path).resolve()


#from util import encode_image
from openai import OpenAI
from configparser import ConfigParser
import json

def build_presentation_prompt(case_text, differential_diagnosis=None, presentation_cfg=None):
    """Build the slideshow-generation prompt.

    presentation_cfg (required dict) selects the version-specific presentation prompt,
    built by the caller from the version's fields in cabot_prompts.py:
      {"style": "split",      "with_ddx_prefix": ..., "without_ddx_prefix": ..., "body": ...}
      {"style": "monolithic", "prompt": ...}   # v1: single prompt with {case}/{model_differential_diagnosis}
    """
    if presentation_cfg is None:
        raise ValueError(
            "presentation_cfg is required; build it from a VersionConfig "
            "(see cabot_prompts.py / run_cabot.py:presentation_cfg_from_version)"
        )
    cfg = presentation_cfg

    if cfg.get("style") == "monolithic":
        # v1 standard presentation: single template formatted with case + ddx
        return (cfg["prompt"]
                .replace("{case}", case_text)
                .replace("{model_differential_diagnosis}", differential_diagnosis or ""))

    # split style (v1.1, vr1): prefix + body + case + optional ddx section
    if differential_diagnosis:
        prompt_prefix = cfg["with_ddx_prefix"]
        differential_section = f"\n\nHere is your differential diagnosis:\n{differential_diagnosis}"
    else:
        prompt_prefix = cfg["without_ddx_prefix"]
        differential_section = ""

    return (
        prompt_prefix +
        cfg["body"] +
        f"\nHere is the case:\n{case_text}" +
        differential_section
    )


def process_references(text, references, base_path):
    """
    Find all figure/table references in text and convert to markdown format
    """
    print(f"[DEBUG] Processing references with base path: {base_path}")
    
    # Resolve the base path - this handles both relative and absolute paths properly
    resolved_base_path = resolve_base_path(base_path)
    
    print(f"[DEBUG] Resolved base path: {resolved_base_path}")
    
    # Create a mapping from base reference numbers to actual references
    # Note: Figure 2A and 2B both map to the same base figure "f2"
    ref_mapping = {}
    
    for ref in references:
        ref_type = ref["type"]
        rid = ref["rid"]
        path = ref["path"]
        
        # Always convert to JPG path - no TIF fallback
        jpg_path = path.replace('article_images', 'article_images_jpg').replace('.tif', '.jpg')
        
        print(f"[DEBUG] Original path: {path}")
        print(f"[DEBUG] Converting to JPG path: {jpg_path}")
        
        # Check if the JPG version exists - warn if not found but continue
        jpg_absolute_path = resolved_base_path / jpg_path
        print(f"[DEBUG] Full JPG absolute path: {jpg_absolute_path}")
        print(f"[DEBUG] JPG path exists: {jpg_absolute_path.exists()}")
        
        if jpg_absolute_path.exists():
            path = jpg_path
            print(f"Using JPG: {path}")
        else:
            print(f"WARNING: JPEG image not found: {jpg_absolute_path} - skipping this image")
            continue  # Skip this reference and continue with the next one
        
        # Create absolute path to the image
        absolute_path = resolved_base_path / path
        
        # Extract base number from rid (e.g., "f1" -> "1", "f2" -> "2")
        if ref_type == "fig":
            match = re.match(r"f(\d+)", rid)
            if match:
                num = match.group(1)
                ref_mapping[f"figure_{num}"] = {
                    "path": str(absolute_path),
                    "rid": rid
                }
        
        elif ref_type == "table":
            match = re.match(r"t(\d+)", rid)
            if match:
                num = match.group(1)
                ref_mapping[f"table_{num}"] = {
                    "path": str(absolute_path),
                    "rid": rid
                }
    
    # Find all figure/table references in the text using regex
    # Pattern matches: Figure/figure/Table/table + number + optional letter
    pattern = r'\b(Figure|figure|Table|table)\s+(\d+)([A-Za-z]?)\b'
    
    def replace_reference(match):
        ref_type = match.group(1)
        number = match.group(2)
        letter = match.group(3).upper() if match.group(3) else ""
        
        # Create the lookup key for the base figure/table number
        lookup_key = f"{ref_type.lower()}_{number}"
        
        # Create display name with the letter if present
        if ref_type.lower() == "figure":
            display_name = f"Figure {number}{letter}"
        else:  # table
            display_name = f"Table {number}{letter}"
        
        # Check if we have this base reference in our mapping
        if lookup_key in ref_mapping:
            ref_info = ref_mapping[lookup_key]
            # Convert to markdown image reference
            return f"![{display_name}]({ref_info['path']})"
        else:
            # If no matching reference found, return original text
            return match.group(0)
    
    # Replace all matches in the text
    processed_text = re.sub(pattern, replace_reference, text)
    
    return processed_text


def parse_llm_output(output: str) -> Tuple[str, Dict[int, str]]:
    """
    Parse the LLM output to extract LaTeX code and spoken transcript.
    
    Returns:
        tuple: (latex_code, transcript_dict) where transcript_dict maps slide numbers to text
    """
    # Extract LaTeX code using the new structured format
    latex_start = "### LATEX_PRESENTATION_START"
    latex_end = "### LATEX_PRESENTATION_END"
    
    latex_start_idx = output.find(latex_start)
    latex_end_idx = output.find(latex_end)
    
    if latex_start_idx != -1 and latex_end_idx != -1:
        latex_section = output[latex_start_idx + len(latex_start):latex_end_idx].strip()
        # Extract code from latex code block
        latex_pattern = r'```latex\s*\n(.*?)\n```'
        latex_match = re.search(latex_pattern, latex_section, re.DOTALL)
        latex_code = latex_match.group(1) if latex_match else latex_section
    else:
        # Fallback to old pattern
        latex_pattern = r'```latex\s*\n(.*?)\n```'
        latex_match = re.search(latex_pattern, output, re.DOTALL)
        latex_code = latex_match.group(1) if latex_match else ""
    
    # Extract spoken transcript using the new structured format
    transcript_start = "### SPOKEN_TRANSCRIPT_START"
    transcript_end = "### SPOKEN_TRANSCRIPT_END"
    
    transcript_start_idx = output.find(transcript_start)
    transcript_end_idx = output.find(transcript_end)
    
    transcript_dict = {}
    
    if transcript_start_idx != -1 and transcript_end_idx != -1:
        transcript_section = output[transcript_start_idx + len(transcript_start):transcript_end_idx].strip()
    else:
        # Fallback to searching the entire output
        transcript_section = output
    
    # Pattern to match [Slide X] followed by content until next [Slide Y] or end
    transcript_pattern = r'\[Slide (\d+)\]\s*\n(.*?)(?=\[Slide \d+\]|###|$)'
    transcript_matches = re.findall(transcript_pattern, transcript_section, re.DOTALL)
    
    for match in transcript_matches:
        slide_num = int(match[0])
        content = match[1].strip()
        transcript_dict[slide_num] = content
    
    return latex_code, transcript_dict


def validate_image_paths(latex_code: str) -> None:
    """
    Validate that all image paths referenced in the LaTeX code exist.
    
    Args:
        latex_code: The LaTeX code to validate
        
    Raises:
        FileNotFoundError: If any referenced image paths don't exist
        ValueError: If no image paths are found in LaTeX when expected
    """
    # Pattern to match \includegraphics commands and extract the path
    pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    
    # Find all image paths in the LaTeX
    image_paths = re.findall(pattern, latex_code)
    
    if not image_paths:
        print("Warning: No image paths found in LaTeX code")
        return
    
    print(f"Validating {len(image_paths)} image paths in LaTeX...")
    
    missing_paths = []
    for image_path in image_paths:
        # Convert to Path object for easier handling
        path = Path(image_path)
        
        # Check if the path exists (could be relative or absolute)
        if not path.exists():
            missing_paths.append(str(path))
        else:
            print(f"Valid path: {path}")
    
    if missing_paths:
        error_msg = f"Invalid image paths found in LaTeX:\n" + "\n".join(f"  - {path}" for path in missing_paths)
        print(f"{error_msg}")
        raise FileNotFoundError(error_msg)
    
    print(f"All {len(image_paths)} image paths are valid")


def adjust_figure_sizes(latex_code: str) -> str:
    """
    Automatically adjust figure sizes based on actual image dimensions to ensure they fit on slides.
    """
    # Pattern to match \includegraphics commands
    pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    
    def replace_includegraphics(match):
        image_path = match.group(1)
        
        try:
            # Open the image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                
                # Calculate appropriate LaTeX sizing based on aspect ratio
                if aspect_ratio > 1.5:  # Wide image
                    # Use width-based sizing for landscape images
                    if aspect_ratio > 2.0:  # Very wide
                        size_spec = "width=0.9\\textwidth,keepaspectratio"
                    else:
                        size_spec = "width=0.8\\textwidth,keepaspectratio"
                elif aspect_ratio < 0.7:  # Tall image
                    # Use height-based sizing for portrait images
                    size_spec = "height=0.6\\textheight,keepaspectratio"
                else:  # Square-ish image
                    size_spec = "width=0.7\\textwidth,keepaspectratio"
                
                return f"\\includegraphics[{size_spec}]{{{image_path}}}"
                
        except Exception as e:
            print(f"Warning: Could not analyze image {image_path}: {e}")
            # Fallback to conservative sizing
            return f"\\includegraphics[width=0.7\\textwidth,keepaspectratio]{{{image_path}}}"
    
    # Replace all \includegraphics commands
    adjusted_latex = re.sub(pattern, replace_includegraphics, latex_code)
    
    # Also handle cases where multiple images might be on the same slide
    # Look for slides with multiple \includegraphics commands
    slides = adjusted_latex.split('\\begin{frame}')
    
    for i, slide in enumerate(slides[1:], 1):  # Skip the preamble
        # Count includegraphics commands in this slide
        img_count = len(re.findall(r'\\includegraphics', slide))
        
        if img_count > 1:
            # Multiple images on one slide - make them smaller
            def make_smaller(match):
                image_path = match.group(1)
                if img_count == 2:
                    size_spec = "width=0.45\\textwidth,keepaspectratio"
                else:  # 3 or more images
                    size_spec = "width=0.3\\textwidth,keepaspectratio"
                return f"\\includegraphics[{size_spec}]{{{image_path}}}"
            
            # Update this slide
            slides[i] = re.sub(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', make_smaller, slide)
    
    # Reconstruct the LaTeX
    if len(slides) > 1:
        adjusted_latex = slides[0] + '\\begin{frame}' + '\\begin{frame}'.join(slides[1:])
    
    return adjusted_latex


def compile_latex_to_pdf(latex_code: str, output_dir: Path, case_id: str, base_cpc_path: str, verbose: bool = True) -> Path:
    """
    Compile LaTeX code to PDF and return the path to the PDF file.
    """
    # Create a temporary directory for LaTeX compilation
    temp_dir = output_dir / "temp_latex"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Write LaTeX code to file
    tex_file = temp_dir / f"{case_id}_presentation.tex"
    
    # No need to fix paths since we're now using absolute paths
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    # Compile LaTeX to PDF using pdflatex
    try:
        # Run pdflatex twice to resolve references
        for _ in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_file.name],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("LaTeX compilation returned nonzero")
                # if verbose:
                #     print(f"LaTeX compilation error: {result.stderr}")
                #     print(f"LaTeX output: {result.stdout}")
    
        pdf_file = temp_dir / f"{case_id}_presentation.pdf"
        if pdf_file.exists():
            # Copy PDF to output directory
            final_pdf = output_dir / f"{case_id}_presentation.pdf"
            shutil.copy(pdf_file, final_pdf)
            return final_pdf
        else:
            raise FileNotFoundError(f"PDF not generated: {pdf_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX: {e}")
        raise
    except FileNotFoundError:
        print("pdflatex not found. Please install LaTeX (e.g., 'brew install mactex' on macOS)")
        raise


def pdf_to_images(pdf_path: Path, output_dir: Path, verbose: bool = True) -> List[Path]:
    """
    Convert PDF slides to individual image files.
    """
    images_dir = output_dir / "slides"
    
    # Clean up any existing images to avoid stale files
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # First, get the number of pages in the PDF
        page_count_result = subprocess.run([
            'pdfinfo', str(pdf_path)
        ], capture_output=True, text=True)
        
        pdf_page_count = None
        if page_count_result.returncode == 0:
            for line in page_count_result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    pdf_page_count = int(line.split(':')[1].strip())
                    break
        
        # Use pdftoppm to convert PDF to images with lower DPI for faster processing
        result = subprocess.run([
            'pdftoppm', 
            '-png', 
            '-r', '300',  # 150 DPI - sufficient for video, much faster than 300 DPI
            str(pdf_path),
            str(images_dir / 'slide')
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"PDF to image conversion error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, 'pdftoppm')
        
        # Get list of generated image files
        image_files = sorted(images_dir.glob('slide-*.png'))
        
        # Validate the count matches expectations
        if pdf_page_count and len(image_files) != pdf_page_count:
            print(f"Warning: PDF has {pdf_page_count} pages but generated {len(image_files)} images")
            print(f"Generated images: {[f.name for f in image_files]}")
        else:
            if verbose:
                print(f"Successfully generated {len(image_files)} images from {pdf_page_count} PDF pages")
        
        return image_files
        
    except FileNotFoundError:
        print("pdftoppm not found. Please install poppler-utils (e.g., 'brew install poppler' on macOS)")
        raise


def _make_tts_call(client: OpenAI, text: str, audio_file: Path) -> None:
    """
    Make the actual TTS call - separated for timeout handling.
    """
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="alloy",
        input=text,
        response_format="mp3"
    ) as response:
        response.stream_to_file(audio_file)


def generate_tts_for_slide(slide_num: int, text: str, audio_file: Path, client: OpenAI, max_retries: int = 3, cancellation_event=None) -> Tuple[int, bool, str]:
    """
    Generate TTS audio for a single slide with retry logic and timeout.
    
    Returns:
        tuple: (slide_num, success, error_message)
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        return (slide_num, False, "TTS generation cancelled")
        
    # Replace CaBot with Cabot for correct pronunciation in TTS
    text = text.replace("CaBot", "Cabot")
    
    for attempt in range(max_retries):
        try:
            # Check for cancellation before each attempt
            if cancellation_event and cancellation_event.is_set():
                return (slide_num, False, "TTS generation cancelled during retry")
                
            print(f"Starting TTS generation for slide {slide_num} (attempt {attempt + 1}/{max_retries})")
            
            # Use ThreadPoolExecutor to wrap the OpenAI call with a timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_make_tts_call, client, text, audio_file)
                try:
                    # Wait for completion with a 15-second timeout (reduced from 30)
                    future.result(timeout=15.0)
                    
                    # Check for cancellation after completion
                    if cancellation_event and cancellation_event.is_set():
                        return (slide_num, False, "TTS generation cancelled after completion")
                        
                    print(f"Generated audio for slide {slide_num}")
                    return (slide_num, True, None)
                    
                except concurrent.futures.TimeoutError:
                    raise TimeoutError("TTS call timed out after 15 seconds")
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            print(f"Error generating TTS for slide {slide_num}: {error_msg}")
            
            if attempt < max_retries - 1:
                print(f"Retrying slide {slide_num}...")
                time.sleep(1)  # Brief delay between retries
            else:
                final_error = f"Failed after {max_retries} attempts: {str(e)}"
                print(f"Failed to generate TTS for slide {slide_num}: {final_error}")
                return (slide_num, False, final_error)
    
    return (slide_num, False, "Unknown error")


def generate_tts_audio(transcript_dict: Dict[int, str], output_dir: Path, client: OpenAI, case_id: str, cancellation_event=None) -> List[Path]:
    """
    Generate TTS audio files for each slide in parallel.
    """
    print(f"Starting TTS generation for {len(transcript_dict)} slides...")
    
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        raise Exception("TTS generation cancelled before starting")
    
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Prepare tasks for parallel execution
    tasks = []
    for slide_num, text in transcript_dict.items():
        audio_file = audio_dir / f"slide_{slide_num:02d}.mp3"
        tasks.append((slide_num, text, audio_file))
        
    if not tasks:
        print("No TTS tasks to process")
        return []
    
    print(f"Generating TTS for {len(tasks)} slides using {len(tasks)} parallel workers...")
    # Limit max workers to avoid too many parallel TTS requests
    max_workers = min(len(tasks), 12)  # OpenAI typically allows up to 50 TPS

    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Check for cancellation before submitting tasks
        if cancellation_event and cancellation_event.is_set():
            raise Exception("TTS generation cancelled before submitting tasks")
            
        # Submit all TTS tasks
        future_to_slide = {
            executor.submit(generate_tts_for_slide, slide_num, text, audio_file, client, 3, cancellation_event): slide_num
            for slide_num, text, audio_file in tasks
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_slide):
            # Check for cancellation during processing
            if cancellation_event and cancellation_event.is_set():
                # Cancel remaining futures
                for f in future_to_slide:
                    if not f.done():
                        f.cancel()
                raise Exception("TTS generation cancelled during processing")
                
            slide_num = future_to_slide[future]
            try:
                # Get result with a timeout (this is in addition to the client timeout)
                result_slide_num, success, error_msg = future.result(timeout=20)  # 20 second total timeout (reduced from 60)
                results[result_slide_num] = (success, error_msg)
                
            except concurrent.futures.TimeoutError:
                error_msg = f"TTS generation timed out after 20 seconds"
                print(f"Timeout error for slide {slide_num}: {error_msg}")
                results[slide_num] = (False, error_msg)
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"Unexpected error for slide {slide_num}: {error_msg}")
                results[slide_num] = (False, error_msg)
    
    # Check for any failures
    failed_slides = []
    successful_slides = []
    
    for slide_num in sorted(transcript_dict.keys()):
        success, error_msg = results.get(slide_num, (False, "No result returned"))
        if success:
            successful_slides.append(slide_num)
        else:
            failed_slides.append((slide_num, error_msg))
    
    if failed_slides:
        failure_details = "; ".join([f"Slide {num}: {error}" for num, error in failed_slides])
        raise RuntimeError(f"TTS generation failed for {len(failed_slides)} slides: {failure_details}")
    
    # Return audio files in the correct order
    audio_files = []
    for slide_num in sorted(transcript_dict.keys()):
        audio_file = audio_dir / f"slide_{slide_num:02d}.mp3"
        audio_files.append(audio_file)
    
    print(f"Successfully generated TTS for all {len(successful_slides)} slides")
    return audio_files


def create_video(image_files: List[Path], audio_files: List[Path], output_dir: Path, case_id: str) -> Path:
    """
    Combine slide images and audio into a video with optimized performance.
    """
    overall_start_time = time.time()
    video_file = output_dir / f"{case_id}_presentation.mp4"
    
    print(f"\n=== STARTING VIDEO CREATION FOR {case_id} ===")
    print(f"Input: {len(image_files)} images, {len(audio_files)} audio files")
    
    # Create a temporary directory for ffmpeg processing
    temp_dir = output_dir / "temp_video"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use original images directly (they're already smaller than 1080p)
        print(f"\n--- PHASE 1: IMAGE PREPARATION ---")
        resize_start_time = time.time()
        print(f"Preparing {len(image_files)} images for video creation...")
        
        # Check original image sizes for reference
        for i, image_file in enumerate(image_files):
            try:
                with Image.open(image_file) as img:
                    orig_size = img.size
                    file_size = image_file.stat().st_size / (1024*1024)
                    print(f"  Image {i+1}: {image_file.name} ({orig_size[0]}x{orig_size[1]}, {file_size:.1f}MB)")
            except Exception as e:
                print(f"  Image {i+1}: {image_file.name} (could not read size: {e})")
        
        # Use original images directly - no resizing needed
        resized_images = list(image_files)
        
        resize_elapsed = time.time() - resize_start_time
        print(f"Image preparation complete: {len(resized_images)} images ready ({resize_elapsed:.1f}s)")
        
        print(f"\n--- PHASE 2: ENCODER SETUP ---")
        hw_detect_start_time = time.time()
        print("Using libx264 encoder (optimized for static slides)")
        hw_detect_elapsed = time.time() - hw_detect_start_time
        print(f"Encoder setup complete ({hw_detect_elapsed:.1f}s)")
        
        # Create video segments in parallel then concatenate
        print(f"\n--- PHASE 3: SEGMENT CREATION ---")
        segment_start_time = time.time()
        total_segments = len(resized_images)
        print(f"Creating {total_segments} video segments sequentially (fast encoding)...")
        
        # Create segments sequentially - threading overhead not worth it with 1fps encoding
        segment_files = []
        total_duration = 0
        
        for i, (image_file, audio_file) in enumerate(zip(resized_images, audio_files)):
            segment_file = temp_dir / f"segment_{i:02d}.mp4"
            
            start_time = time.time()
            print(f"  Creating segment {i+1}/{total_segments} (slide {i+1})")
            
            try:
                # 1. Measure the MP3 a little more accurately
                duration = float(subprocess.check_output([
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=duration',
                    '-of', 'default=nw=1:nk=1',
                    str(audio_file)
                ]).strip())

                # 2. Force the video stream to the *next* full second (+1 s safety)
                video_len = math.ceil(duration) + 1        # e.g. 6.84 -> 8 s

                segment_cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1', '-r', '1', '-i', str(image_file),   # stay at 1 fps
                    '-i', str(audio_file),
                    # video
                    '-c:v', 'libx264', '-preset', 'veryfast',
                    '-tune', 'stillimage', '-crf', '25',
                    # audio
                    '-c:a', 'aac', '-b:a', '128k',
                    '-pix_fmt', 'yuv420p',
                    '-t', str(video_len),     # ← make video a little longer …
                    '-shortest',              # … then chop when audio finishes
                    str(segment_file)
                ]
                
                print(f"    Encoding with libx264...")
                result = subprocess.run(segment_cmd, capture_output=True, text=True)
                
                elapsed = time.time() - start_time
                if result.returncode == 0:
                    file_size = segment_file.stat().st_size / (1024*1024) if segment_file.exists() else 0
                    print(f"    Completed in {elapsed:.1f}s ({file_size:.1f}MB)")
                    segment_files.append(segment_file)
                    total_duration += duration
                else:
                    print(f"    Failed after {elapsed:.1f}s")
                    print(f"    Error: {result.stderr}")
                    raise RuntimeError(f"Segment {i+1} failed: {result.stderr}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    Failed after {elapsed:.1f}s with exception: {e}")
                raise RuntimeError(f"Segment {i+1} failed: {e}")
        
        if not segment_files:
            raise RuntimeError("No video segments were created successfully")
        
        print(f"\nAll {len(segment_files)} segments created successfully (total duration: {total_duration:.1f}s)")
        
        segment_elapsed = time.time() - segment_start_time
        print(f"Segment creation complete ({segment_elapsed:.1f}s)")
        print(f"Preparing to concatenate {len(segment_files)} segments (total duration: {total_duration:.1f}s)")
        
        print(f"\n--- PHASE 4: CONCATENATION ---")
        # Concatenate all segments
        concat_file = temp_dir / "concat_list.txt"
        print(f"Creating concatenation file: {concat_file}")
        
        with open(concat_file, 'w') as f:
            for i, segment_file in enumerate(segment_files):
                f.write(f"file '{segment_file.absolute()}'\n")
                print(f"  Adding to concat: {segment_file.name}")
        
        # Show the concatenation file contents for debugging
        print(f"\nConcatenation file contents:")
        with open(concat_file, 'r') as f:
            content = f.read()
            print(content)
        
        # Final concatenation with optimized settings
        print(f"Starting final concatenation of {len(segment_files)} segments...")
        concat_start_time = time.time()
        
        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_file), 
            '-c', 'copy',  # Copy streams without re-encoding
            str(video_file)
        ]
        
        print(f"Concatenation command: {' '.join(concat_cmd)}")
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        
        concat_elapsed = time.time() - concat_start_time
        
        if result.returncode != 0:
            print(f"Concatenation failed after {concat_elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            print(f"Output: {result.stdout}")
            raise subprocess.CalledProcessError(result.returncode, 'ffmpeg')
        
        # Verify final video
        if video_file.exists():
            final_size = video_file.stat().st_size / (1024*1024)
            
            # Get final video duration
            try:
                duration_result = subprocess.run([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(video_file)
                ], capture_output=True, text=True, timeout=10)
                
                if duration_result.returncode == 0:
                    final_duration = float(duration_result.stdout.strip())
                    print(f"Final video duration: {final_duration:.1f}s (expected: {total_duration:.1f}s)")
                    
                    if abs(final_duration - total_duration) > 2.0:  # Allow 2s tolerance
                        print(f"WARNING: Duration mismatch! Expected {total_duration:.1f}s, got {final_duration:.1f}s")
                else:
                    print(f"Could not get final video duration: {duration_result.stderr}")
                    
            except Exception as e:
                print(f"Error checking final video duration: {e}")
            
            print(f"Video created successfully in {concat_elapsed:.1f}s: {video_file}")
            print(f"  Final video size: {final_size:.1f}MB")
            
            # Overall timing summary
            overall_elapsed = time.time() - overall_start_time
            print(f"\n=== VIDEO CREATION SUMMARY ===")
            print(f"Total time: {overall_elapsed:.1f}s")
            print(f"  Phase 1 - Image preparation: {resize_elapsed:.1f}s ({resize_elapsed/overall_elapsed*100:.1f}%)")
            print(f"  Phase 2 - Hardware detection: {hw_detect_elapsed:.1f}s ({hw_detect_elapsed/overall_elapsed*100:.1f}%)")
            print(f"  Phase 3 - Segment creation: {segment_elapsed:.1f}s ({segment_elapsed/overall_elapsed*100:.1f}%)")
            print(f"  Phase 4 - Concatenation: {concat_elapsed:.1f}s ({concat_elapsed/overall_elapsed*100:.1f}%)")
            print(f"Segments processed: {len(segment_files)}/{total_segments}")
            print(f"Expected vs actual duration: {total_duration:.1f}s vs {final_duration:.1f}s")
            print(f"================================")
        else:
            print(f"Video file not created: {video_file}")
            raise FileNotFoundError(f"Video file not created: {video_file}")
        
        return video_file
        
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg (e.g., 'sudo apt install ffmpeg' on Ubuntu)")
        raise
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def generate_video_for_case(
        case, 
        base_differential_diagnosis,
        base_cpc_path, 
        api_key, 
        output_base_dir, 
        max_retries=3,
        progress_callback=None,
        base_model="o3",
        cancellation_event=None,
        presentation_cfg=None
        ):
    """
    Generate a video presentation for a single case with retry logic.
    
    Args:
        case: Case data dictionary
        base_cpc_path: Base path for CPC files
        api_key: OpenAI API key
        output_base_dir: Base directory for output files
        max_retries: Maximum number of retry attempts
        progress_callback: Optional callback function to report progress
        base_model: Base model to use for generation
        cancellation_event: Optional threading.Event to signal cancellation
    
    Returns:
        tuple: (case_id, success, error_message)
    """
    case_id = case["id"]
    
    def report_progress(stage, status, progress=0, video_path=None):
        """Helper function to report progress if callback is provided"""
        if progress_callback:
            # Check if progress callback signals cancellation
            should_continue = progress_callback(case_id, stage, status, progress, video_path)
            if should_continue is False:
                raise Exception(f"Video generation cancelled during {stage}")
        
        # Also check cancellation event directly
        if cancellation_event and cancellation_event.is_set():
            raise Exception(f"Video generation cancelled during {stage}")
    
    for attempt in range(max_retries):
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                raise Exception("Video generation cancelled before processing")
                
            print(f"Processing case {case_id} (attempt {attempt + 1}/{max_retries})")
            report_progress("initialization", "Starting video generation", 0)
            
            # Create OpenAI client for this process
            client = OpenAI(api_key=api_key)
            
            # Create output directory
            output_dir = Path(output_base_dir) / case_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the presentation of case text
            poc = case["presentation_of_case"]
            references = case["presentation_of_case_references"]
            
            # Debug: Print what references we received
            print(f"[DEBUG] Video generation for case {case_id}")
            print(f"[DEBUG] Received {len(references)} image references:")
            for i, ref in enumerate(references):
                print(f"[DEBUG] Reference {i}: type={ref.get('type')}, rid={ref.get('rid')}, path={ref.get('path')}")
            
            # Resolve the base path properly (same logic as in process_references)
            resolved_base_path = resolve_base_path(base_cpc_path)
            
            print(f"[DEBUG] Resolved base path for image processing: {resolved_base_path}")
            
            processed_poc = process_references(poc, references, base_cpc_path)

            # Generate LLM prompt using conditional prompt builder
            prompt = build_presentation_prompt(
                case_text=processed_poc,
                differential_diagnosis=base_differential_diagnosis,
                presentation_cfg=presentation_cfg
            )

            # Prepare messages with images
            messages = [
                {
                   "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ]
            
            for image in references:
                # Check for cancellation during image processing
                if cancellation_event and cancellation_event.is_set():
                    raise Exception("Video generation cancelled during image processing")
                    
                # Always convert to JPG path - no TIF fallback
                print(f"[DEBUG] Original image data: {image}")
                print(f"[DEBUG] Original image path: {image['path']}")
                
                jpg_path = image['path'].replace('article_images', 'article_images_jpg').replace('.tif', '.jpg')
                print(f"[DEBUG] Converted JPG path: {jpg_path}")
                
                image_path = resolved_base_path / jpg_path
                
                print(f"[DEBUG] Processing image: {image}")
                print(f"[DEBUG] Converting to JPG path: {jpg_path}")
                print(f"[DEBUG] Full image path: {image_path}")
                print(f"[DEBUG] Path exists: {image_path.exists()}")
                
                if not image_path.exists():
                    print(f"WARNING: JPEG image not found: {image_path} - skipping this image for LLM processing")
                    continue  # Skip this image and continue with the next one
                
                try:
                    print(f"[DEBUG] About to call encode_image with: {str(image_path)}")
                    base64_image = encode_image(str(image_path))
                    
                    # Use the JPG path for LaTeX
                    absolute_path = image_path
                    
                    print(f"[DEBUG] Absolute path for LaTeX: {absolute_path}")
                    
                    # Add caption text with figure/table number AND the exact path to use in LaTeX
                    if image["type"] == "table":
                        num = int(image["rid"].lstrip('t'))
                        caption_text = f"Here is Table {num}"
                    elif image["type"] == "fig":
                        num = int(image["rid"].lstrip('f'))
                        caption_text = f"Here is Figure {num}. In your LaTeX presentation, use this exact path for \\includegraphics: {absolute_path}"
                    else:
                        caption_text = f"Here is {image['type']} {image['rid']}. In your LaTeX presentation, use this exact path for \\includegraphics: {absolute_path}"
                    
                    print(f"[DEBUG] Caption text: {caption_text[:100]}...")
                    
                    # Create separate user message block for each image, like in cabot2.py
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": caption_text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    })
                    print(f"[DEBUG] Successfully added image to messages")
                except Exception as e:
                    print(f"WARNING: Could not encode image {image_path}: {e} - skipping this image")
                    continue  # Skip this image and continue with the next one
            
            # Call LLM
            # Check for cancellation before LLM call
            if cancellation_event and cancellation_event.is_set():
                raise Exception("Video generation cancelled before LLM generation")
                
            print(f"Calling LLM for case {case_id}...")
            report_progress("llm_generation", "Generating transcript with AI", 20)
            model_kwargs = {}
            if base_model == "o3":
                model_kwargs = {"reasoning": {"effort": "high"}}

            response = call_with_retry(
                client.responses.create,
                model=base_model,
                #model="o4-mini",
                input=messages,
                **model_kwargs
            )
            
            # Check for cancellation after LLM call
            if cancellation_event and cancellation_event.is_set():
                raise Exception("Video generation cancelled after LLM generation")
            
            output = response.output_text
            report_progress("llm_generation", "Transcript generated successfully", 40)
            
            # Save the raw LLM output for debugging
            with open(output_dir / "llm_output.txt", "w", encoding="utf-8") as f:
                f.write(output)
            
            print(f"[DEBUG] LLM output length: {len(output)} characters")
            includegraphics_search = "\\includegraphics"
            print(f"[DEBUG] LLM output contains '{includegraphics_search}': {includegraphics_search in output}")
            
            # Parse LLM output
            latex_code, transcript_dict = parse_llm_output(output)
            print(f"Parsed {len(transcript_dict)} slides from transcript for case {case_id}")
            
            if not latex_code:
                raise ValueError("No LaTeX code found in LLM output")
            
            if not transcript_dict:
                raise ValueError("No transcript found in LLM output")
            
            print(f"[DEBUG] LaTeX code length: {len(latex_code)} characters")
            print(f"[DEBUG] LaTeX contains '{includegraphics_search}': {includegraphics_search in latex_code}")
            
            # Save original LaTeX code for debugging
            with open(output_dir / "presentation_original.tex", "w", encoding="utf-8") as f:
                f.write(latex_code)
            
            # Validate that all image paths in the LaTeX exist
            print(f"Validating image paths in LaTeX for case {case_id}...")
            validate_image_paths(latex_code)
            
            # Automatically adjust figure sizes based on image dimensions
            print(f"Analyzing and adjusting figure sizes for case {case_id}...")
            adjusted_latex_code = adjust_figure_sizes(latex_code)
            
            # Save adjusted LaTeX code
            with open(output_dir / "presentation.tex", "w", encoding="utf-8") as f:
                f.write(adjusted_latex_code)
            
            # Compile LaTeX to PDF
            print(f"Compiling LaTeX to PDF for case {case_id}...")
            pdf_path = compile_latex_to_pdf(adjusted_latex_code, output_dir, case_id, base_cpc_path)
            print(f"PDF created for case {case_id}: {pdf_path}")
            
            # Convert PDF to images
            print(f"Converting PDF to slide images for case {case_id}...")
            image_files = pdf_to_images(pdf_path, output_dir)
            print(f"Generated {len(image_files)} slide images for case {case_id}")
            
            # Validate slide count matches transcript entries before generating TTS
            num_slides = len(image_files)
            num_transcript_entries = len(transcript_dict)
            if num_slides != num_transcript_entries:
                raise ValueError(f"Slide count mismatch: {num_slides} slides in PDF vs {num_transcript_entries} transcript entries. "
                               f"Expected transcript entries for slides: {sorted(range(1, num_slides + 1))}, "
                               f"Found transcript entries for slides: {sorted(transcript_dict.keys())}")
            
            print(f"Validation passed: {num_slides} slides match {num_transcript_entries} transcript entries")
            
            # Generate TTS audio
            # Check for cancellation before TTS generation
            if cancellation_event and cancellation_event.is_set():
                raise Exception("Video generation cancelled before TTS generation")
                
            print(f"Generating TTS audio for slides for case {case_id}...")
            report_progress("tts_generation", "Generating text-to-speech audio", 50)
            audio_files = generate_tts_audio(transcript_dict, output_dir, client, case_id, cancellation_event)
            print(f"Generated {len(audio_files)} audio files for case {case_id}")
            report_progress("tts_generation", "Audio generation completed", 70)
            
            # Create video (lengths are guaranteed to match due to validation above)
            # Check for cancellation before video creation
            if cancellation_event and cancellation_event.is_set():
                raise Exception("Video generation cancelled before video creation")
                
            print(f"Creating final video for case {case_id}...")
            report_progress("video_creation", "Preparing final video", 80)
            video_path = create_video(image_files, audio_files, output_dir, case_id)
            print(f"Video presentation created for case {case_id}: {video_path}")
            
            # Report final progress (job worker will handle completion status update)
            report_progress("video_creation", "Video generation completed successfully", 100)
            return (case_id, True, None)
                
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            print(f"Error processing case {case_id}: {error_msg}")
            
            if attempt < max_retries - 1:
                # This is an intermediate failure - report as retry, not final error
                report_progress("retry_error", f"Attempt {attempt + 1} failed, retrying: {str(e)}", 0)
                print(f"Retrying case {case_id} in 5 seconds...")
                time.sleep(5)
            else:
                # This is the final failure after all retries - report as error
                print(f"All {max_retries} attempts failed for case {case_id}")
                report_progress("error", f"Video generation failed after {max_retries} attempts: {str(e)}", 0)
                return (case_id, False, error_msg)
    
    # This should never happen, but if it does, report it as a final error
    print(f"Unexpected code path: All {max_retries} attempts completed without returning for case {case_id}")
    report_progress("error", f"Video generation failed: Unknown error after {max_retries} attempts", 0)
    return (case_id, False, "Unknown error")


def process_case_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments.
    """
    return generate_video_for_case(*args)

# TODO: fix the function calling here. We don't correctly bass base_differential_diagnosis, so this will not work
def generate_videos_parallel(cases, base_cpc_path, api_key, output_base_dir, max_workers=None, max_retries=3):
    """
    Generate videos for multiple cases in parallel using multiprocessing.
    
    Args:
        cases: List of case data dictionaries
        base_cpc_path: Base path for CPC files
        api_key: OpenAI API key
        output_base_dir: Base directory for output files
        max_workers: Maximum number of worker processes (None for CPU count)
        max_retries: Maximum number of retry attempts per case
    
    Returns:
        dict: Results dictionary with case_id as key and (success, error_message) as value
    """
    if max_workers is None:
        max_workers = min(len(cases), multiprocessing.cpu_count())
    
    print(f"Starting parallel video generation for {len(cases)} cases with {max_workers} workers")
    
    # Prepare arguments for each case
    args_list = [
        (case, base_cpc_path, api_key, output_base_dir, max_retries)
        for case in cases
    ]
    
    results = {}
    
    with multiprocessing.Pool(max_workers) as pool:
        # Process cases in parallel
        pool_results = pool.map(process_case_wrapper, args_list)
        
        # Collect results
        for case_id, success, error_msg in pool_results:
            results[case_id] = (success, error_msg)
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate video presentations from medical cases")
    parser.add_argument("--base_cpc_path", type=str, default="datasets/nejm_cpcs",
                        help="Base path for CPC files (default: datasets/nejm_cpcs)")
    parser.add_argument("--output_base_dir", type=str, default="output_testing_2",
                        help="Base directory for output files (default: output_testing_2)")
    parser.add_argument("--case_path", type=str, default="benchmarks/datasets/sample_ddx_cases_website_7_16_25.json",
                        help="Path to the JSON file containing case data (default: benchmarks/datasets/sample_ddx_cases_website_7_16_25.json)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Maximum number of worker processes (default: 8)")
    
    args = parser.parse_args()
    
    base_cpc_path = args.base_cpc_path
    output_base_dir = args.output_base_dir
 
    config = ConfigParser()
    config.read("config.ini")

    api_key = config.get("main", "OPENAI_KEY_TB")

    with open(args.case_path) as f:
        target_cases = json.load(f)

    # Then, among those target cases, filter out ones that already have videos
    cases = []
    for case in target_cases:
        case_id = case['id']
        
        # Check if video already exists for this case
        output_dir = Path(output_base_dir) / case_id
        video_file = output_dir / f"{case_id}_presentation.mp4"
        
        if video_file.exists():
            print(f"Skipping {case_id} - video already exists")
        else:
            cases.append(case)

    print(f"Processing {len(cases)} most recent cases:")
    for i, case in enumerate(cases, 1):
        print(f"  {i}. {case['id']} ({case['publication_date']})")
   
    # Generate videos in parallel
    start_time = time.time()
    results = generate_videos_parallel(
        cases=cases,
        base_cpc_path=base_cpc_path,
        api_key=api_key,
        output_base_dir=output_base_dir,
        max_workers=args.max_workers,
        max_retries=3
    )
    end_time = time.time()
    
    # Print results summary
    print(f"\nVideo generation completed in {end_time - start_time:.2f} seconds")
    print("\nResults summary:")
    
    successful_cases = []
    failed_cases = []
    
    for case_id, (success, error_msg) in results.items():
        if success:
            successful_cases.append(case_id)
            print(f"{case_id}: SUCCESS")
        else:
            failed_cases.append(case_id)
            print(f"{case_id}: FAILED - {error_msg}")
    
    print(f"\nOverall: {len(successful_cases)}/{len(cases)} cases completed successfully")
    
    if failed_cases:
        print(f"Failed cases: {', '.join(failed_cases)}")






