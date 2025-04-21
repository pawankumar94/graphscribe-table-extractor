#!/usr/bin/env python3
"""
Document extraction script for processing PDFs and images with charts/diagrams.
This script accepts a file path and extracts structured data as Markdown.
"""

import os
import argparse
from pathlib import Path
import sys
from extract import DocumentProcessor

def process_file(input_path, output_path=None):
    """Process a document and save the extracted data as markdown.
    
    Args:
        input_path: Path to the input document (PDF, image)
        output_path: Path to save the output Markdown file
    """
    # Validate input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    # Validate file type
    file_ext = Path(input_path).suffix.lower()
    if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: PDF, PNG, JPG, JPEG")
        sys.exit(1)
    
    # Create processor
    processor = DocumentProcessor()
    
    # Generate output path if not provided
    if not output_path:
        input_file = Path(input_path)
        output_path = os.path.join(
            os.path.dirname(input_file),
            f"{input_file.stem}_extracted.md"
        )
    
    print(f"Processing file: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Process the document
        markdown_output = processor.process_document(input_path, output_path)
        
        print(f"\nSuccess! Data extracted and saved to: {output_path}")
        print("\nPreview of extracted content:")
        print("="*50)
        preview = markdown_output[:500] + "..." if len(markdown_output) > 500 else markdown_output
        print(preview)
        print("="*50)
        return True
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract data from documents with charts/diagrams and convert to Markdown"
    )
    parser.add_argument(
        "input_path",
        help="Path to the input document (PDF, PNG, JPG)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output Markdown file (default: <input_name>_extracted.md)",
        default=None
    )
    
    args = parser.parse_args()
    process_file(args.input_path, args.output)

if __name__ == "__main__":
    main() 