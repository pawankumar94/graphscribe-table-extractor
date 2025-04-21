#!/usr/bin/env python3
"""
PDF to Markdown Converter

This script converts a PDF document to a Markdown document while preserving text
structure and converting charts/diagrams to Markdown tables.

Usage:
    python pdf_to_markdown.py input.pdf [--output output.md] [--cleanup]
"""

import os
import sys
import argparse
from pathlib import Path
from extract import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("--output", "-o", help="Path to the output Markdown file (optional)")
    parser.add_argument("--cleanup", "-c", action="store_true", help="Clean up temporary files after processing")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_pdf)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.pdf':
        print(f"Error: Input file must be a PDF document.")
        sys.exit(1)
    
    # Set default output path if not provided
    output_path = args.output
    if not output_path:
        output_path = input_path.with_suffix('.md')
    
    # Create the document processor
    processor = DocumentProcessor()
    
    print(f"Converting PDF: {input_path}")
    print(f"Output Markdown: {output_path}")
    
    try:
        # Process the document
        markdown_content = processor.process_document(
            str(input_path),
            output_path=str(output_path),
            cleanup=args.cleanup
        )
        
        print(f"Conversion complete! Markdown saved to {output_path}")
        
        if args.cleanup:
            print("Temporary files have been cleaned up.")
        else:
            print("Temporary files were preserved for reference.")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 