"""
Utility functions for document processing and extraction.
"""

import os
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Any, Tuple
import tempfile

def extract_images_from_pdf_pymupdf(pdf_path: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
    """Extract images from a PDF file using PyMuPDF with metadata.
    
    This function extracts images with higher quality and provides metadata about position.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples containing (PIL Image, metadata dict)
    """
    result = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num, page in enumerate(pdf_document):
        # Get images
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get position data
            rect = page.get_image_bbox(img)
            
            # Load as PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create metadata
            metadata = {
                "page_num": page_num + 1,
                "img_index": img_index,
                "width": image.width,
                "height": image.height,
                "position": {
                    "x1": rect.x0,
                    "y1": rect.y0,
                    "x2": rect.x1,
                    "y2": rect.y1,
                },
                "image_format": base_image["ext"]
            }
            
            result.append((image, metadata))
    
    pdf_document.close()
    return result

def save_temp_image(image: Image.Image, format: str = "PNG") -> str:
    """Save a PIL image to a temporary file and return the path.
    
    Args:
        image: PIL Image object
        format: Image format to save as
        
    Returns:
        Path to the saved temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format.lower()}")
    image.save(temp_file.name, format=format)
    temp_file.close()
    return temp_file.name

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")

def detect_chart_type(image: Image.Image) -> str:
    """Detect the type of chart in an image.
    
    This is a placeholder function. In a real implementation, you might use
    a separate ML model or additional LLM queries to detect the chart type.
    
    Args:
        image: PIL Image object
        
    Returns:
        String describing the chart type
    """
    # This would be implemented with actual detection logic
    return "unknown_chart" 