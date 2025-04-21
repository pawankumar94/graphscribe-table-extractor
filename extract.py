import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import io
import google.generativeai as genai
from langchain.agents import AgentExecutor, initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import time
import threading
import pandas as pd
import traceback

# Try to import img2table
try:
    from img2table.document import PDF, Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except ImportError:
    print("img2table not available. Please install it using: pip install img2table")
    IMG2TABLE_AVAILABLE = False

# Load environment variables (API keys)
load_dotenv()

# Configure Google GenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY environment variable")
genai.configure(api_key=GOOGLE_API_KEY)

def has_graph(page, min_paths=20, min_img_area=10000):
    """
    Detects if a PDF page contains a graph by checking for drawing paths or large images.
    
    Args:
        page: The PyMuPDF page object
        min_paths (int): Minimum number of drawing paths to indicate a graph
        min_img_area (int): Minimum image area (width * height) to indicate a graph
        
    Returns:
        bool: True if a graph is detected, False otherwise
    """
    # Count drawing paths (lines, curves for vector graphs)
    paths = page.get_drawings()
    if len(paths) > min_paths:
        return True
        
    # Check for images (rasterized graphs)
    images = page.get_images(full=True)
    for img in images:
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            if base_image and "width" in base_image and "height" in base_image:
                if base_image["width"] * base_image["height"] > min_img_area:
                    return True
        except Exception as e:
            print(f"Error checking image: {e}")
            
    return False

class DocumentProcessor:
    """Main processor for extracting data from documents with diagrams/charts."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-001"):
        """Initialize the document processor.
        
        Args:
            model_name: The Gemini model to use
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)
        self.setup_agent()
    
    def setup_agent(self):
        """Set up the agent with specific tools."""
        # Define tools using the Tool wrapper
        tools = [
            Tool(
                name="extract_text_from_pdf",
                func=self.extract_text_from_pdf,
                description="Extract text from a PDF file."
            ),
            Tool(
                name="analyze_chart_image",
                func=self.analyze_chart_image,
                description="Analyze a chart/diagram image and convert it to a Markdown table. Input is the path to an image file."
            ),
            Tool(
                name="extract_tables_from_pdf",
                func=self.extract_tables_from_pdf,
                description="Extract tables from a PDF file using img2table. Returns data in markdown format."
            ),
            Tool(
                name="extract_charts_from_pdf",
                func=self.extract_charts_from_pdf,
                description="Extract charts and diagrams from a PDF file. Returns paths to extracted chart images."
            ),
            Tool(
                name="process_image_batch",
                func=self.process_image_batch,
                description="Process a batch of chart/diagram image files. Input is a list of image file paths. Returns markdown results."
            )
        ]
        
        # Initialize the agent with a higher iteration limit
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15  # Increased to handle more complex documents
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def extract_tables_from_pdf(self, pdf_path: str) -> str:
        """Extract tables from a PDF file using img2table.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown string with extracted tables
        """
        if not IMG2TABLE_AVAILABLE:
            return "Error: img2table is not installed. Please install it with: pip install img2table"
        
        try:
            # Create the PDF object with img2table
            pdf_doc = PDF(pdf_path)
            
            # Use Tesseract OCR for extraction
            ocr = TesseractOCR(n_threads=1, lang="eng")
            
            # Extract tables
            extracted_tables = pdf_doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=True,
                min_confidence=50
            )
            
            # Convert to markdown
            markdown_result = "# Extracted Tables\n\n"
            
            for page_idx, tables in extracted_tables.items():
                if tables:
                    markdown_result += f"## Page {page_idx + 1}\n\n"
                    for i, table in enumerate(tables):
                        markdown_result += f"### Table {i + 1}\n\n"
                        # Get pandas dataframe and convert to markdown
                        df = table.df
                        markdown_result += df.to_markdown(index=False) + "\n\n"
            
            return markdown_result
        
        except Exception as e:
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            return f"Error extracting tables from PDF: {error_msg}\n\n{traceback_str}"
    
    def extract_charts_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract charts and diagrams from a PDF file.
        
        Uses both PyMuPDF detection and img2table extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to extracted chart images
        """
        # Create directory for extracted images
        pdf_path_obj = Path(pdf_path)
        pdf_name = pdf_path_obj.stem
        
        # Create target directory for charts
        target_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "extracted_images" / pdf_name / "charts"
        os.makedirs(target_dir, exist_ok=True)
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        chart_paths = []
        
        # Extract charts using PyMuPDF detection
        for page_idx, page in enumerate(doc):
            # Check if page has a graph
            if has_graph(page, min_paths=15, min_img_area=8000):
                print(f"Found chart on page {page_idx+1}")
                
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_path = str(target_dir / f"chart_p{page_idx+1}.png")
                pix.save(img_path)
                chart_paths.append(img_path)
                
            # Extract standalone images that might be charts
            images = page.get_images(full=True)
            if images:
                print(f"Found {len(images)} images on page {page_idx+1}")
                for img_idx, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            # Check if image is large enough to be a chart
                            if base_image.get("width", 0) * base_image.get("height", 0) > 40000:
                                # Save to a file in the target directory
                                img_path = str(target_dir / f"image_p{page_idx+1}_{img_idx+1}.{base_image['ext']}")
                                with open(img_path, "wb") as f:
                                    f.write(base_image["image"])
                                chart_paths.append(img_path)
                    except Exception as e:
                        print(f"Error extracting image: {str(e)}")
        
        doc.close()
        
        # If no charts were found via PyMuPDF, try using img2table
        if not chart_paths and IMG2TABLE_AVAILABLE:
            try:
                pdf_doc = PDF(pdf_path)
                ocr = TesseractOCR(n_threads=1, lang="eng")
                
                # Extract tables (some charts might be recognized as tables)
                extracted_tables = pdf_doc.extract_tables(
                    ocr=ocr,
                    implicit_rows=True,
                    implicit_columns=True,
                    borderless_tables=True,
                    min_confidence=50
                )
                
                # Save each extracted table as an image
                for page_idx, tables in extracted_tables.items():
                    if tables:
                        images = convert_from_path(pdf_path, first_page=page_idx+1, last_page=page_idx+1)
                        if images:
                            page_image = images[0]
                            for i, table in enumerate(tables):
                                # Get bounding box
                                bbox = table.bbox
                                # Crop the image to the table region
                                crop = page_image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                                # Save the cropped image
                                img_path = str(target_dir / f"table_p{page_idx+1}_{i+1}.png")
                                crop.save(img_path)
                                chart_paths.append(img_path)
            except Exception as e:
                print(f"Error using img2table: {str(e)}")
        
        return chart_paths
    
    def process_image_batch(self, image_paths: List[str]) -> str:
        """Process a batch of chart/diagram images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Markdown results from processing the images
        """
        # Input validation: ensure image_paths is a proper list of strings
        if isinstance(image_paths, str):
            # If a string was passed instead of a list, try to parse it
            print(f"Warning: Expected a list of image paths but got a string: {image_paths}")
            try:
                # Try to interpret as a comma-separated list
                if ',' in image_paths:
                    image_paths = [path.strip() for path in image_paths.split(',')]
                # Or as a single path
                else:
                    image_paths = [image_paths]
            except Exception as e:
                print(f"Error parsing image paths: {e}")
                return "Error: Invalid input format. Expected a list of image paths."
        
        # Further validate each path
        valid_paths = []
        for path in image_paths:
            if isinstance(path, str) and os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Skipping invalid path: {path}")
        
        if not valid_paths:
            return "No valid image paths found to process."
        
        batch_size = 3  # Process max 3 images per batch to stay within quota
        results = []
        
        print(f"Processing {len(valid_paths)} images in batches of {batch_size}...")
        
        for i in range(0, len(valid_paths), batch_size):
            batch = valid_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(valid_paths) + batch_size - 1)//batch_size}...")
            
            batch_results = []
            for img_path in batch:
                try:
                    print(f"Analyzing image: {os.path.basename(img_path)}")
                    
                    # Try using img2table first for tables
                    if IMG2TABLE_AVAILABLE:
                        try:
                            img_doc = Img2TableImage(img_path)
                            ocr = TesseractOCR(n_threads=1, lang="eng")
                            tables = img_doc.extract_tables(
                                ocr=ocr,
                                implicit_rows=True,
                                implicit_columns=True,
                                borderless_tables=True,
                                min_confidence=50
                            )
                            
                            if tables:
                                table_output = f"## Extracted Table: {os.path.basename(img_path)}\n\n"
                                for i, table in enumerate(tables):
                                    table_output += table.df.to_markdown(index=False) + "\n\n"
                                batch_results.append(table_output)
                                continue
                        except Exception as e:
                            print(f"img2table failed: {e}, falling back to Gemini")
                    
                    # Fall back to Gemini analysis
                    result = self.analyze_chart_image(img_path)
                    # Add the image name to the result
                    image_name = os.path.basename(img_path)
                    chart_type = result.split("\n")[0].replace("##", "").strip()
                    result = f"## {chart_type} from {image_name}\n\n" + "\n".join(result.split("\n")[1:])
                    batch_results.append(result)
                    
                    # Add a small delay to avoid hitting rate limits
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error analyzing image: {str(e)}")
                    batch_results.append(f"## Error\nFailed to analyze image {os.path.basename(img_path)}: {str(e)}")
            
            results.extend(batch_results)
            
            # Add a delay between batches to respect API rate limits
            if i + batch_size < len(valid_paths):
                print("Pausing between batches to respect API rate limits...")
                time.sleep(2)
        
        return "\n\n".join(results)
    
    def analyze_chart_image(self, image_path: str) -> str:
        """Analyze a chart/diagram image and convert it to a Markdown table.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Markdown table representation of the chart data
        """
        MAX_RETRIES = 2
        retry_count = 0
        
        # Validate the image path
        if not os.path.exists(image_path):
            return f"## Error\n\nImage file not found: {image_path}"
            
        # Get the file extension
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in ['.png', '.jpg', '.jpeg']:
            # Try to convert to PNG if it's not one of the supported formats
            try:
                print(f"Converting {file_ext} image to PNG format...")
                img = Image.open(image_path)
                png_path = os.path.splitext(image_path)[0] + ".png"
                img.save(png_path, "PNG")
                image_path = png_path
                print(f"Converted image saved to {png_path}")
            except Exception as e:
                print(f"Failed to convert image: {e}")
                # Continue with the original file
        
        while retry_count <= MAX_RETRIES:
            try:
                # Read the image file
                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                except Exception as file_error:
                    return f"## Error\n\nFailed to read image file: {str(file_error)}"
                
                # Check if image is empty or too small
                if len(image_bytes) < 1000:  # Smaller than 1KB
                    return "## Empty Image\n\nThe image is too small or empty to analyze."
                
                # First, analyze the chart to get a description
                model = genai.GenerativeModel('gemini-2.0-flash-001')
                
                # Use a timeout for the API call
                description_result = [None]
                description_error = [None]
                
                def call_api():
                    try:
                        response = model.generate_content([
                            "Analyze this image which contains a chart/graph/diagram. Describe in detail what it shows, including all data points, axes, labels, and values. Be comprehensive and precise with all numeric values.",
                            {"mime_type": "image/png", "data": image_bytes}
                        ])
                        description_result[0] = response.text
                    except Exception as e:
                        description_error[0] = str(e)
                
                # Start the API call in a thread with a timeout
                thread = threading.Thread(target=call_api)
                thread.start()
                thread.join(timeout=20)  # 20 second timeout
                
                if thread.is_alive():
                    # API call is taking too long, consider it failed
                    return "## Timeout Error\n\nThe image analysis took too long to complete."
                
                if description_error[0]:
                    # Handle specific error types
                    error_msg = description_error[0]
                    if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                        if retry_count < MAX_RETRIES:
                            print(f"API rate limit hit, retrying in 5 seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
                            time.sleep(5)
                            retry_count += 1
                            continue
                        return "## Rate Limit Error\n\nGoogle AI API rate limit exceeded. Please try again later."
                    
                    # Handle image format errors - try with a different prompt
                    if "image" in error_msg.lower() and "format" in error_msg.lower():
                        try:
                            # Try with a different approach - convert image first
                            img = Image.open(image_path)
                            img_bytes_io = io.BytesIO()
                            img.save(img_bytes_io, format='PNG')
                            image_bytes = img_bytes_io.getvalue()
                            
                            # Try again with the converted image
                            response = model.generate_content([
                                "Analyze this image which contains a chart/graph/diagram.",
                                {"mime_type": "image/png", "data": image_bytes}
                            ])
                            description_result[0] = response.text
                        except Exception as e:
                            return f"## Error\n\nFailed to process image after conversion: {str(e)}"
                    else:
                        return f"## Error\n\nError analyzing image: {error_msg}"
                
                if not description_result[0]:
                    return "## Error\n\nFailed to generate a description for the image."
                
                # Then generate a Markdown table from the description
                table_result = [None]
                table_error = [None]
                
                def call_table_api():
                    try:
                        response = model.generate_content(
                            f"""Based on this chart description, create a properly formatted Markdown table that accurately represents the data:
                            {description_result[0]}
                            
                            The table should:
                            1. Have clear column headers
                            2. Include ALL data points and numeric values from the description
                            3. Be neatly formatted in valid Markdown syntax
                            4. Preserve the numerical relationships in the chart
                            5. Use pipes and dashes to format the table properly (proper markdown format)
                            
                            IMPORTANT: Extract ALL numeric values from the description and include them in the table.
                            IMPORTANT: Format the table as a proper markdown table with headers.
                            
                            Only return the Markdown table, nothing else."""
                        )
                        table_result[0] = response.text
                    except Exception as e:
                        table_error[0] = str(e)
                
                # Start the API call in a thread with a timeout
                thread = threading.Thread(target=call_table_api)
                thread.start()
                thread.join(timeout=15)  # 15 second timeout
                
                if thread.is_alive():
                    # API call is taking too long, return just the description
                    return f"## Data Table\n\nTable generation timed out."
                
                if table_error[0]:
                    # Try a second approach to table generation with a different prompt
                    try:
                        response = model.generate_content(
                            f"""You are tasked with converting a chart description into a markdown table.
                            
                            Description of the chart:
                            {description_result[0]}
                            
                            Create a markdown table that captures ALL the numeric data points from this description.
                            The table must:
                            1. Have appropriate column headers based on the chart axes
                            2. Include every data point mentioned in the description
                            3. Be formatted with proper markdown table syntax using | and -
                            4. Only contain the table, no additional text
                            
                            Format the table with proper markdown headers and rows."""
                        )
                        table_result[0] = response.text
                    except Exception as second_error:
                        # Return at least the description if table generation fails twice
                        return f"## Data Table\n\nFailed to generate table: {table_error[0]}"
                
                if not table_result[0]:
                    # Return just the description if table is empty
                    return f"## Data Table\n\nNo data table could be generated from this image."
                
                # Check if the result contains a properly formatted markdown table
                if '|' not in table_result[0] or '-' not in table_result[0]:
                    # Try to force the model to format it correctly
                    try:
                        response = model.generate_content(
                            f"""The following data needs to be formatted as a proper Markdown table with headers:
                            
                            {table_result[0]}
                            
                            Convert it to a Markdown table that uses | and - characters for formatting.
                            Make sure the table has headers and rows properly aligned.
                            Only return the formatted Markdown table, nothing else."""
                        )
                        reformatted_table = response.text
                        if '|' in reformatted_table and '-' in reformatted_table:
                            table_result[0] = reformatted_table
                    except Exception as format_error:
                        # Keep the original result if reformatting fails
                        pass
                
                # Return only the table, with a minimal header
                chart_type = self._determine_chart_type(description_result[0])
                
                return f"## {chart_type} Data\n\n{table_result[0]}"
                
            except Exception as e:
                if retry_count < MAX_RETRIES:
                    print(f"Error occurred, retrying in 3 seconds... (attempt {retry_count + 1}/{MAX_RETRIES})")
                    time.sleep(3)
                    retry_count += 1
                else:
                    error_msg = str(e)
                    return f"## Error\n\nFailed to analyze image after {MAX_RETRIES} attempts: {error_msg}"
        
        # This should not be reached, but just in case
        return "## Error\n\nUnexpected error during image analysis."
        
    def _determine_chart_type(self, description):
        """Determine the chart type from the description."""
        chart_types = {
            "bar chart": ["bar chart", "bar graph", "column chart", "histogram"],
            "line chart": ["line chart", "line graph", "time series"],
            "pie chart": ["pie chart", "donut chart", "circular chart"],
            "scatter plot": ["scatter plot", "scatter chart", "scatter diagram"],
            "area chart": ["area chart", "stacked area"],
            "combination chart": ["combination chart", "mixed chart", "dual axis"],
        }
        
        description_lower = description.lower()
        
        for chart_type, keywords in chart_types.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return chart_type.title()
        
        return "Chart"

    def process_document(self, file_path: str, output_path: Optional[str] = None, cleanup: bool = False) -> str:
        """Process a document and extract tabular data from diagrams.
        
        Args:
            file_path: Path to the document (PDF, image)
            output_path: Path to save the output Markdown file
            cleanup: Whether to clean up temporary files after processing
            
        Returns:
            Markdown string with extracted tables
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            # Handle different file types
            if file_extension == '.pdf':
                # Extract text directly from PDF first
                print(f"Extracting text from {file_path}")
                text_content = self.extract_text_from_pdf(str(file_path))
                
                # Extract charts first - we'll need the paths
                print(f"Extracting charts from {file_path}")
                chart_paths = self.extract_charts_from_pdf(str(file_path))
                
                # Extract tables
                print(f"Extracting tables from {file_path}")
                table_results = self.extract_tables_from_pdf(str(file_path))

                # Process charts to get their analysis
                chart_analyses = {}
                if chart_paths:
                    print(f"Found {len(chart_paths)} charts/diagrams, processing them...")
                    for i, chart_path in enumerate(chart_paths):
                        print(f"Analyzing chart {i+1}/{len(chart_paths)}: {os.path.basename(chart_path)}")
                        result = self.analyze_chart_image(chart_path)
                        chart_analyses[chart_path] = result
                
                # Create a proper Markdown document
                md_content = f"# {file_path.stem}\n\n"
                
                # Add sections from the text content
                lines = text_content.split('\n')
                current_section = []
                sections = []
                
                for line in lines:
                    line = line.strip()
                    # Check if line starts a new section (e.g., all caps or numbered)
                    if (line.isupper() and len(line) > 3) or \
                    (line.startswith(tuple("123456789")) and line.find(".") < 3):
                        if current_section:
                            sections.append('\n'.join(current_section))
                            current_section = []
                        current_section.append(f"## {line}")
                    else:
                        current_section.append(line)
                
                # Add the last section
                if current_section:
                    sections.append('\n'.join(current_section))
                
                # Extract tables by page
                tables_by_page = {}
                lines = table_results.split('\n')
                current_page = 0
                current_table_content = []
                in_table = False
                
                for line in lines:
                    if line.startswith("## Page "):
                        try:
                            current_page = int(line.replace("## Page ", "").strip())
                        except ValueError:
                            current_page = 0
                        continue
                    
                    # Check if we're in a table
                    if line.startswith("|") or line.startswith("### Table"):
                        in_table = True
                        current_table_content.append(line)
                    elif in_table and not line.strip():
                        # End of table
                        if current_page not in tables_by_page:
                            tables_by_page[current_page] = []
                        tables_by_page[current_page].append('\n'.join(current_table_content))
                        current_table_content = []
                        in_table = False
                
                # Combine text content with tables and chart analyses
                section_count = len(sections)
                charts_added = set()
                
                # Simple heuristic: distribute tables and charts proportionally through the document
                if section_count > 0:
                    for i, section in enumerate(sections):
                        md_content += section + "\n\n"
                        
                        # Add tables and charts based on proportion through the document
                        section_proportion = (i + 1) / section_count
                        page_proportion = int(section_proportion * len(tables_by_page))
                        
                        # Add tables for this section
                        for page in range(page_proportion - 1, page_proportion + 1):
                            if page in tables_by_page:
                                for table in tables_by_page[page]:
                                    md_content += table + "\n\n"
                        
                        # Add chart analyses for this section if available
                        if chart_paths:
                            charts_for_section = [path for path in chart_paths if path not in charts_added]
                            if charts_for_section:
                                # Add a proportional number of charts to this section
                                num_charts_to_add = max(1, int(len(charts_for_section) / (section_count - i))) if section_count > i else len(charts_for_section)
                                for j in range(min(num_charts_to_add, len(charts_for_section))):
                                    chart_path = charts_for_section[j]
                                    image_name = os.path.basename(chart_path)
                                    chart_analysis = chart_analyses[chart_path]
                                    chart_type = chart_analysis.split("\n")[0].replace("##", "").strip()
                                    
                                    # Add a minimal header with the chart name
                                    md_content += f"### {chart_type} from {image_name}\n\n"
                                    
                                    # Add only the table part
                                    md_content += chart_analysis.split("\n\n", 1)[1] + "\n\n"
                                    charts_added.add(chart_path)
                else:
                    # Simple document with no clear sections, just append everything
                    md_content += text_content + "\n\n"
                    
                    # Add tables
                    for page in sorted(tables_by_page.keys()):
                        for table in tables_by_page[page]:
                            md_content += table + "\n\n"
                    
                    # Add chart analyses
                    for chart_path in chart_paths:
                        image_name = os.path.basename(chart_path)
                        chart_analysis = chart_analyses[chart_path]
                        chart_type = chart_analysis.split("\n")[0].replace("##", "").strip()
                        
                        # Add a minimal header with the chart name
                        md_content += f"### {chart_type} from {image_name}\n\n"
                        
                        # Add only the table part
                        md_content += chart_analysis.split("\n\n", 1)[1] + "\n\n"
                
                # Add any remaining charts that weren't added
                remaining_charts = [path for path in chart_paths if path not in charts_added]
                if remaining_charts:
                    md_content += "## Additional Charts and Diagrams\n\n"
                    for chart_path in remaining_charts:
                        image_name = os.path.basename(chart_path)
                        chart_analysis = chart_analyses[chart_path]
                        chart_type = chart_analysis.split("\n")[0].replace("##", "").strip()
                        
                        # Add a minimal header with the chart name
                        md_content += f"### {chart_type} from {image_name}\n\n"
                        
                        # Add only the table part
                        md_content += chart_analysis.split("\n\n", 1)[1] + "\n\n"
            
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                # For single images, use img2table first, then fallback to Gemini
                if IMG2TABLE_AVAILABLE:
                    try:
                        img_doc = Img2TableImage(str(file_path))
                        ocr = TesseractOCR(n_threads=1, lang="eng")
                        tables = img_doc.extract_tables(
                            ocr=ocr,
                            implicit_rows=True,
                            implicit_columns=True,
                            borderless_tables=True,
                            min_confidence=50
                        )
                        
                        if tables:
                            result = f"# Table extracted from {file_path.name}\n\n"
                            for i, table in enumerate(tables):
                                result += f"## Table {i+1}\n\n"
                                result += table.df.to_markdown(index=False) + "\n\n"
                            md_content = result
                        else:
                            # No tables found, try Gemini analysis
                            result = self.analyze_chart_image(str(file_path))
                            image_name = file_path.name
                            chart_type = result.split("\n")[0].replace("##", "").strip()
                            
                            # Format the result with minimal description
                            table_content = result.split("\n\n", 1)[1] if "\n\n" in result else result
                            md_content = f"# Extracted Data from {file_path.name}\n\n### {chart_type} from {image_name}\n\n{table_content}"
                    except Exception as e:
                        print(f"img2table failed: {e}, falling back to Gemini")
                        result = self.analyze_chart_image(str(file_path))
                        image_name = file_path.name
                        chart_type = result.split("\n")[0].replace("##", "").strip()
                        
                        # Format the result with minimal description
                        table_content = result.split("\n\n", 1)[1] if "\n\n" in result else result
                        md_content = f"# Extracted Data from {file_path.name}\n\n### {chart_type} from {image_name}\n\n{table_content}"
                else:
                    # img2table not available, use Gemini directly
                    result = self.analyze_chart_image(str(file_path))
                    image_name = file_path.name
                    chart_type = result.split("\n")[0].replace("##", "").strip()
                    
                    # Format the result with minimal description
                    table_content = result.split("\n\n", 1)[1] if "\n\n" in result else result
                    md_content = f"# Extracted Data from {file_path.name}\n\n### {chart_type} from {image_name}\n\n{table_content}"
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Save to file if output_path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(md_content)
            
            return md_content
            
        finally:
            # Clean up temporary files if cleanup is True
            # This ensures cleanup is performed even if an error occurs
            if cleanup:
                print("Cleaning up temporary files...")
                self._cleanup_temp_files(file_path)
        
    def _cleanup_temp_files(self, file_path=None):
        """Clean up temporary files created during processing.
        
        Args:
            file_path: Path to the document (to help identify related temp files)
        """
        # If no specific file path is provided, we can't do targeted cleanup
        if not file_path:
            return
            
        file_path = Path(file_path)
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Clean up extracted images directory
        if file_path.stem:
            extracted_dir = script_dir / "extracted_images" / file_path.stem
            if extracted_dir.exists():
                import shutil
                try:
                    shutil.rmtree(extracted_dir)
                    print(f"Cleaned up: {extracted_dir}")
                except Exception as e:
                    print(f"Failed to clean up {extracted_dir}: {e}")
            
            # Clean up analysis files
            analysis_dir = script_dir / f"{file_path.stem}_analysis"
            if analysis_dir.exists():
                import shutil
                try:
                    shutil.rmtree(analysis_dir)
                    print(f"Cleaned up: {analysis_dir}")
                except Exception as e:
                    print(f"Failed to clean up {analysis_dir}: {e}")
                    
            # Clean up any temporary PNG files
            png_path = file_path.parent / f"{file_path.stem}.png"
            if png_path.exists():
                try:
                    os.remove(png_path)
                    print(f"Cleaned up: {png_path}")
                except Exception as e:
                    print(f"Failed to clean up {png_path}: {e}")
                
            # Clean up any temporary converted images
            temp_files = list(file_path.parent.glob(f"{file_path.stem}_*.png"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    print(f"Cleaned up: {temp_file}")
                except Exception as e:
                    print(f"Failed to clean up {temp_file}: {e}")

    def _get_extracted_images_dir(self, file_path):
        """Get the path to the extracted images directory for a file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Path to the extracted images directory
        """
        file_path = Path(file_path)
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        extracted_dir = script_dir / "extracted_images" / file_path.stem
        return str(extracted_dir) 