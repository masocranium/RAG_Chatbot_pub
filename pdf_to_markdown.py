from abc import ABC, abstractmethod
import glob
import os
from pathlib import Path
import time
import io

from pdf2image import convert_from_path
import fitz  # PyMuPDF
import base64
from google import genai
from google.genai import types
from PIL import Image
import pytesseract

# Set Tesseract path for Windows if the script won't work otherwise. Uncomment and set the correct path if needed.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Interfaces (Strategy Pattern)

class PresentationExtractor(ABC):
    """Strategy for extracting content from a presentation."""
    @abstractmethod
    def extract_slides(self, file_path: str) -> list[bytes]:
        pass

class SlideAnalyzer(ABC):
    """Strategy for analyzing a single slide."""
    @abstractmethod
    def analyze_slide(self, slide_data: bytes, page_num: int) -> str:
        pass

class ErrorHandlingStrategy(ABC):
    """Strategy for handling execution errors."""
    @abstractmethod
    def execute(self, operation, *args, **kwargs):
        pass

# 2. Concrete Implementations

class ExponentialBackoffStrategy(ErrorHandlingStrategy):
    """Retries operations with an exponentially increasing delay."""
    def __init__(self, max_retries: int = 5, base_delay: int = 2):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute(self, operation, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                # Check for rate limiting or 503 unavailable
                if attempt == self.max_retries - 1:
                    print(f"Operation failed after {self.max_retries} attempts. Giving up.")
                    raise
                
                delay = self.base_delay * (2 ** attempt)
                print(f"API error: {e}. Retrying in {delay} seconds (Attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(delay)

class PDFToImageExtractor(PresentationExtractor):
    """Extracts PDF pages as images using PyMuPDF (requires pip install pymupdf). No Poppler needed!"""
    
    def extract_slides(self, file_path: str) -> list[bytes]:
        print(f"Extracting images from {file_path} using PyMuPDF...")
        doc = fitz.open(file_path)
        slide_bytes = []
        for page in doc:
            # Render page to an image pixmap
            pix = page.get_pixmap(dpi=200)
            # Convert directly to PNG bytes
            slide_bytes.append(pix.tobytes("png"))
        return slide_bytes

class GeminiSlideAgent(SlideAnalyzer):
    """Agent that uses Gemini to convert a slide image to Markdown."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert technical writer. Convert the provided presentation slide into highly structured, detailed Markdown. "
        "Include all headers, bullet points, and text. Describe any important diagrams, charts, or images in detail using markdown syntax."
    )

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-lite",
        error_handler: ErrorHandlingStrategy = None,
        system_prompt: str = None,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.error_handler = error_handler or ExponentialBackoffStrategy()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def analyze_slide(self, slide_data: bytes, page_num: int) -> str:
        print(f"Agent is analyzing slide {page_num}...")
        
        contents = [
            types.Part.from_bytes(data=slide_data, mime_type="image/png"),
            f"{self.system_prompt}\n\nPlease parse slide {page_num}."
        ]
        
        def _call_api():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            return response.text
            
        return self.error_handler.execute(_call_api)

class TesseractSlideAgent(SlideAnalyzer):
    """Agent that uses Tesseract OCR to extract text from a slide image."""
    
    def __init__(self, tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def analyze_slide(self, slide_data: bytes, page_num: int) -> str:
        print(f"Agent is analyzing slide {page_num} using Tesseract OCR...")
        image = Image.open(io.BytesIO(slide_data))
        text = pytesseract.image_to_string(image)
        
        # Return as basic Markdown text block
        return f"**Extracted Text (OCR):**\n\n{text.strip()}"

# 3. The Orchestrator (Pipeline Pattern)

SLIDE_IMAGE_LONG_EDGE = 1280  # px cap for the copy that gets attached to generation calls


class SlideToMarkdownPipeline:
    """Orchestrates the agentic workflow."""

    def __init__(self, extractor: PresentationExtractor, analyzer: SlideAnalyzer):
        self.extractor = extractor
        self.analyzer = analyzer

    def _save_slide_image(self, slide_bytes: bytes, images_dir: str, pdf_stem: str, page_num: int) -> str:
        """Persist the full-res PNG plus a downscaled copy for LLM attachment.

        Returns the downscaled copy's path, relative to `images_dir`'s parent
        (so it stays portable if the repo/db gets moved to another machine).
        """
        slide_dir = os.path.join(images_dir, pdf_stem)
        os.makedirs(slide_dir, exist_ok=True)

        archival_path = os.path.join(slide_dir, f"slide_{page_num:03d}.png")
        with open(archival_path, "wb") as f:
            f.write(slide_bytes)

        attach_path = os.path.join(slide_dir, f"slide_{page_num:03d}_small.png")
        image = Image.open(io.BytesIO(slide_bytes))
        image.thumbnail((SLIDE_IMAGE_LONG_EDGE, SLIDE_IMAGE_LONG_EDGE), Image.LANCZOS)
        image.save(attach_path, format="PNG")

        base_dir = os.path.dirname(os.path.normpath(images_dir))
        return os.path.relpath(attach_path, base_dir).replace(os.sep, "/")
    
    def describe(self) -> str:
        return f"SlideToMarkdownPipeline with extractor: {self.extractor.__class__.__name__}, analyzer: {self.analyzer.__class__.__name__}"

    def process(self, pdf_path: str, output_path: str, overwrite: bool = False, images_dir: str = None):
        if not overwrite and os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            return

        # Step 1: Extract
        slides = self.extractor.extract_slides(pdf_path)

        markdown_output = [f"# Presentation: {Path(pdf_path).name}\n"]
        pdf_stem = Path(pdf_path).stem

        # Step 2: Analyze (Agentic loop)
        for i, slide_bytes in enumerate(slides, start=1):
            slide_md = self.analyzer.analyze_slide(slide_bytes, i)

            # Step 3: Aggregate
            if images_dir:
                image_path = self._save_slide_image(slide_bytes, images_dir, pdf_stem, i)
                markdown_output.append(f'<!-- SLIDE_BOUNDARY id={i} image="{image_path}" -->')
            markdown_output.append(f"## Slide {i}\n")
            markdown_output.append(slide_md)
            markdown_output.append("\n---\n") # Slide separator

        # Step 4: Write Output
        final_markdown = "\n".join(markdown_output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        print(f"Successfully saved Markdown to {output_path}")


def preprocess_pdfs_to_markdown(
    pdf_dir: str,
    output_dir: str,
    api_key: str,
    system_prompt: str = None,
    only_new_files: bool = False,
    images_dir: str = None,
    model_name: str = "gemini-3.1-flash-lite",
) -> str:
    """Convert source PDFs to markdown using the slide pipeline.

    Shared implementation used by presets (e.g. ExampleBot). Pass
    ``system_prompt`` to customize how each slide is converted to markdown;
    defaults to ``GeminiSlideAgent.DEFAULT_SYSTEM_PROMPT``. Pass ``images_dir``
    to also persist each slide's rendered image and embed a machine-readable
    ``SLIDE_BOUNDARY`` marker pointing at it, so a diagram-aware vector DB
    populator can later re-attach the real image at answer time.

    Set ``only_new_files=True`` to skip PDFs that already have an output file
    named ``output_<pdf_stem>.md`` in ``output_dir``.

    Returns the directory containing the generated markdown files.
    """
    os.makedirs(output_dir, exist_ok=True)
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)
    extractor = PDFToImageExtractor()
    error_strategy = ExponentialBackoffStrategy(max_retries=5, base_delay=2)
    analyzer = GeminiSlideAgent(
        api_key=api_key,
        model_name=model_name,
        error_handler=error_strategy,
        system_prompt=system_prompt,
    )
    pipeline = SlideToMarkdownPipeline(extractor=extractor, analyzer=analyzer)

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
    for pdf_path in pdf_paths:
        output_file = os.path.join(output_dir, f"output_{Path(pdf_path).stem}.md")
        pipeline.process(pdf_path, output_file, overwrite=not only_new_files, images_dir=images_dir)

    return output_dir


# Command-line usage: python pdf_to_markdown.py <input_dir> <output_dir>
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown.")
    parser.add_argument("input_dir", help="Directory containing source PDFs.")
    parser.add_argument("output_dir", help="Directory to write Markdown files to.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    args = parser.parse_args()

    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Compose the pipeline with our specific strategies
    extractor = PDFToImageExtractor()
    error_strategy = ExponentialBackoffStrategy(max_retries=5, base_delay=2)


    analyzer = GeminiSlideAgent(api_key=api_key,model_name='gemini-3.1-flash-lite', error_handler=error_strategy)
    # analyzer = TesseractSlideAgent()  # Uncomment to use Tesseract instead of Gemini
    
    pipeline = SlideToMarkdownPipeline(extractor=extractor, analyzer=analyzer)
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    print(pipeline.describe())
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        for file in glob.glob(f"{input_dir}/*.pdf", recursive=True):
            output_file = f"{output_dir}/output_{Path(file).stem}.md"
            pipeline.process(file, output_file, overwrite=args.overwrite)
    except FileNotFoundError as fnfe:
        print(f"File not found: {fnfe}")
    except Exception as e:
        print(f"An error occurred: {e}")
