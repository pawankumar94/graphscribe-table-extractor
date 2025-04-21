#!/usr/bin/env python3
"""
Streamlit app for Graphscribe - Chart-to-Table PDF Extractor

This app allows users to upload documents (PDF, images) and extract tables from charts
using the img2table and Gemini-powered extraction pipeline.
"""

import os
import streamlit as st
import fitz  # PyMuPDF
from pathlib import Path
import tempfile
import time
from PIL import Image
import base64
import io
import json
from extract import DocumentProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="Graphscribe",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Graphscribe - Extract structured tables from charts and diagrams in documents"
    }
)

# Apply custom CSS for light theme styling
st.markdown("""
<style>
    /* Global theme - light color palette */
    body {
        background-color: #ffffff;
        color: #333333;
    }
    
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 0.5rem;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #3b82f6;
        font-weight: 600;
    }
    
    p, li, span {
        color: #333333;
    }
    
    a {
        color: #3b82f6;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #2563eb;
        border: none;
    }
    
    /* Containers */
    .comparison-container {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f8fafc;
    }
    
    .chart-title {
        font-weight: bold;
        color: #3b82f6;
        margin-bottom: 10px;
    }
    
    .table-container {
        padding: 15px;
        background-color: #f8fafc;
        border-radius: 5px;
        border: 1px solid #e2e8f0;
        color: #333333;
    }
    
    /* Expanders */
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: #f8fafc;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Success message */
    .success-message {
        font-weight: bold;
        color: #10b981;
        padding: 15px;
        border-radius: 5px;
        background-color: #ecfdf5;
        margin: 15px 0;
        border: 1px solid #10b981;
    }
    
    /* File uploader */
    .file-uploader {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f8fafc;
    }
    
    /* Sidebar */
    .sidebar-content {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
    }
    
    /* Page navigation */
    .page-navigation {
        margin: 15px 0;
        text-align: center;
    }
    
    .page-button {
        display: inline-block;
        margin: 5px;
        padding: 8px 12px;
        background-color: #f8fafc;
        color: #3b82f6;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        border: 1px solid #3b82f6;
    }
    
    .page-button:hover {
        background-color: #3b82f6;
        color: white;
    }
    
    .page-button.active {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 30px;
    }
    
    .logo-icon {
        margin-right: 10px;
        background-color: #f8fafc;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .logo-text {
        font-size: 24px;
        font-weight: 600;
        color: #3b82f6;
    }
    
    /* PDF viewer styling */
    .pdf-viewer {
        width: 100%;
        height: 800px;
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Document preview header */
    .document-preview-header {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .document-preview-icon {
        margin-right: 10px;
        color: #3b82f6;
    }
    
    .document-preview-title {
        font-size: 1.5em;
        font-weight: 600;
        color: #3b82f6;
    }
    
    /* Processing section */
    .processing-section {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #e2e8f0;
    }
    
    /* Results section */
    .results-section {
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Initialize session state variables if they don't exist
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'pdf_preview' not in st.session_state:
    st.session_state.pdf_preview = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'charts_found' not in st.session_state:
    st.session_state.charts_found = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "upload"
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'total_pages' not in st.session_state:
    st.session_state.total_pages = 0

def display_pdf(file_path):
    """Display a PDF file using an iframe for a professional look"""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" class="pdf-viewer" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_image_base64(image_path):
    """Convert image to base64 string for embedding in HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def display_pdf_page(pdf_path, page_num):
    """Display a PDF page as an image"""
    try:
        doc = fitz.open(pdf_path)
        if page_num < len(doc):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            return Image.open(io.BytesIO(img_bytes))
        else:
            st.error(f"Invalid page number: {page_num}")
            return None
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk and return the path"""
    # Clean up previous file if it exists
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.remove(st.session_state.temp_file_path)
        except:
            pass
    
    # Save new file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.temp_file_path = file_path
    return file_path

def render_page_buttons(total_pages, current_page):
    """Render clickable page buttons for PDF navigation"""
    st.markdown('<div class="page-navigation">', unsafe_allow_html=True)
    
    # Create a cleaner navigation with just prev/next and a page selector
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # Add previous button
        prev_disabled = current_page <= 0
        if prev_disabled:
            st.markdown('<button class="page-button" style="opacity: 0.5; cursor: not-allowed; background-color: #f1f5f9; color: #9ca3af; border-color: #d1d5db;" disabled>◀ Previous</button>', unsafe_allow_html=True)
        else:
            st.markdown(f'<button class="page-button" onclick="document.dispatchEvent(new CustomEvent(\'streamlit:prevPage\'))" style="background-color: #3b82f6; color: white;">◀ Previous</button>', unsafe_allow_html=True)
    
    with col2:
        # Add a page selector dropdown instead of numbered buttons
        if total_pages > 1:
            page_options = [f"Page {i+1} of {total_pages}" for i in range(total_pages)]
            selected_index = st.selectbox("", options=range(total_pages), 
                                         format_func=lambda x: page_options[x],
                                         index=current_page)
            
            # Handle page change from dropdown
            if selected_index != current_page:
                st.session_state.current_page = selected_index
                new_params = st.query_params.to_dict()
                new_params["page"] = selected_index
                st.query_params.update(new_params)
                st.rerun()
    
    with col3:
        # Add next button
        next_disabled = current_page >= total_pages - 1
        if next_disabled:
            st.markdown('<button class="page-button" style="opacity: 0.5; cursor: not-allowed; background-color: #f1f5f9; color: #9ca3af; border-color: #d1d5db;" disabled>Next ▶</button>', unsafe_allow_html=True)
        else:
            st.markdown(f'<button class="page-button" onclick="document.dispatchEvent(new CustomEvent(\'streamlit:nextPage\'))" style="background-color: #3b82f6; color: white;">Next ▶</button>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle page button clicks with JavaScript
    st.markdown("""
    <script>
        document.addEventListener('streamlit:prevPage', function() {
            const cur = Number(window.CurrentPage || 0);
            if (cur > 0) {
                window.CurrentPage = cur - 1;
                window.location.href = window.location.href.split('?')[0] + `?page=${cur - 1}`;
            }
        });
        
        document.addEventListener('streamlit:nextPage', function() {
            const cur = Number(window.CurrentPage || 0);
            const max = Number(window.TotalPages || 1);
            if (cur < max - 1) {
                window.CurrentPage = cur + 1;
                window.location.href = window.location.href.split('?')[0] + `?page=${cur + 1}`;
            }
        });
        
        // Set global variables
        window.CurrentPage = %s;
        window.TotalPages = %s;
    </script>
    """ % (st.session_state.current_page, st.session_state.total_pages), unsafe_allow_html=True)

def clear_submit():
    """Clear the submit state"""
    if "process_clicked" in st.session_state:
        st.session_state.process_clicked = False

def upload_interface():
    """Main upload and preview interface"""
    # Logo and app title header
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">
            <svg width="32" height="32" viewBox="0 0 24 24">
                <rect x="3" y="3" width="4" height="18" fill="#3b82f6"/>
                <rect x="10" y="8" width="4" height="13" fill="#10b981"/>
                <rect x="17" y="5" width="4" height="16" fill="#ef4444"/>
            </svg>
        </div>
        <div class="logo-text">Graphscribe</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.uploaded_file is not None:
        # Document preview section
        st.markdown("""
        <div class="document-preview-header">
            <div class="document-preview-icon">🔍</div>
            <div class="document-preview-title">Document Preview</div>
        </div>
        """, unsafe_allow_html=True)
        
        file_path = st.session_state.temp_file_path
        
        if st.session_state.uploaded_file.type == "application/pdf":
            # Handle PDF preview with modern UI
            try:
                doc = fitz.open(file_path)
                total_pages = len(doc)
                st.session_state.total_pages = total_pages
                
                # Show total pages indicator
                st.markdown(f"<p style='text-align: right; color: #64748b;'>Total pages: {total_pages}</p>", unsafe_allow_html=True)
                
                # Initialize PDF preview pages if needed
                st.session_state.pdf_preview = []
                
                # Page navigation with clickable buttons
                if total_pages > 0:
                    # Check for page parameter in URL - using st.query_params instead of deprecated API
                    if "page" in st.query_params:
                        try:
                            requested_page = int(st.query_params["page"])
                            if 0 <= requested_page < total_pages:
                                st.session_state.current_page = requested_page
                        except ValueError:
                            pass
                    
                    # Display PDF with iframe for professional look
                    display_pdf(file_path)
                    
                    # Render page navigation buttons
                    render_page_buttons(total_pages, st.session_state.current_page)
                    
                doc.close()
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
        else:
            # Handle image preview
            try:
                img = Image.open(file_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        # Process document section
        st.markdown("""
        <div class="processing-section">
            <h3 style="margin-top: 0;">Extract Tables from Charts</h3>
            <p>Process this document to extract structured data from charts and diagrams.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Process Document", key="process_btn", use_container_width=True, on_click=clear_submit):
                st.session_state.process_clicked = True
        
        if "process_clicked" in st.session_state and st.session_state.process_clicked:
            with st.spinner("🔍 Analyzing document and extracting tables from charts..."):
                try:
                    # Create output file path
                    output_filename = f"{os.path.splitext(st.session_state.uploaded_file.name)[0]}_output.md"
                    output_path = os.path.join("output", output_filename)
                    
                    # Process the document with cleanup disabled to keep temp files
                    result = st.session_state.processor.process_document(
                        file_path, 
                        output_path=output_path,
                        cleanup=False  # Keep temporary files for visualization
                    )
                    
                    # Store the extracted data
                    st.session_state.extracted_data = result
                    
                    # Get extracted chart paths to display
                    extracted_dir = Path(st.session_state.processor._get_extracted_images_dir(file_path))
                    charts_dir = extracted_dir / "charts" if extracted_dir.exists() else None
                    
                    if charts_dir and charts_dir.exists():
                        chart_paths = list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.jpg")) + list(charts_dir.glob("*.jpeg"))
                        st.session_state.charts_found = [str(p) for p in chart_paths]
                    else:
                        st.session_state.charts_found = []
                    
                    st.session_state.processing_complete = True
                    
                    # Switch to results tab
                    st.session_state.active_tab = "results"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                
                # Reset the process_clicked state
                st.session_state.process_clicked = False
    else:
        # If no file is uploaded yet, show welcome message
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <h2>Welcome to Graphscribe</h2>
            <p>Upload a document from the sidebar to get started</p>
            <div style="font-size: 5em; margin: 30px 0;">📊</div>
            <p>Extract structured tables from charts and diagrams in your documents</p>
        </div>
        """, unsafe_allow_html=True)

def results_tab():
    """Tab for displaying extraction results"""
    # Logo and app title header
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">
            <svg width="32" height="32" viewBox="0 0 24 24">
                <rect x="3" y="3" width="4" height="18" fill="#3b82f6"/>
                <rect x="10" y="8" width="4" height="13" fill="#10b981"/>
                <rect x="17" y="5" width="4" height="16" fill="#ef4444"/>
            </svg>
        </div>
        <div class="logo-text">Graphscribe</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2>🔍 Extraction Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.extracted_data is None:
        st.warning("No document has been processed yet. Please upload and process a document first.")
        if st.button("⬅️ Back to Upload", use_container_width=True):
            st.session_state.active_tab = "upload"
            st.rerun()
        return
    
    # Success message
    if st.session_state.processing_complete:
        st.markdown(f"""
        <div class="success-message">
            ✅ Successfully processed {st.session_state.uploaded_file.name} and extracted data from {len(st.session_state.charts_found)} charts/diagrams!
        </div>
        """, unsafe_allow_html=True)
        st.session_state.processing_complete = False  # Reset for next time
    
    # Create a download button for the Markdown
    output_filename = f"{os.path.splitext(st.session_state.uploaded_file.name)[0]}_output.md"
    output_path = os.path.join("output", output_filename)
    
    with open(output_path, "r") as f:
        markdown_content = f.read()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create a download button
        st.download_button(
            label="📥 Download Markdown",
            data=markdown_content,
            file_name=output_filename,
            mime="text/markdown",
            use_container_width=True
        )
    
    # Display comparison between original and extracted
    st.markdown("""
    <div class="results-section">
        <h3 style="color: #3b82f6;">📊 Side-by-Side Comparison</h3>
        <p>Compare the original charts with the extracted tabular data:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # If charts were found, show a comparison of each chart with its extracted table
    if st.session_state.charts_found:
        # Find the charts and corresponding tables in the markdown
        chart_sections = markdown_content.split("###")
        
        for i, chart_path in enumerate(st.session_state.charts_found):
            chart_filename = os.path.basename(chart_path)
            
            # Create expandable section for each chart
            with st.expander(f"Chart {i+1}: {chart_filename}", expanded=i==0):
                st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
                cols = st.columns(2)
                
                # Original chart image
                with cols[0]:
                    st.markdown('<p class="chart-title">Original Chart</p>', unsafe_allow_html=True)
                    try:
                        img = Image.open(chart_path)
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                
                # Extracted table
                with cols[1]:
                    st.markdown('<p class="chart-title">Extracted Table</p>', unsafe_allow_html=True)
                    
                    # Find the corresponding section in the markdown
                    table_section = None
                    for section in chart_sections:
                        if chart_filename in section:
                            table_section = section
                            break
                    
                    if table_section:
                        # Extract just the table part
                        table_lines = []
                        capture = False
                        for line in table_section.split("\n"):
                            if line.strip().startswith("```") and not capture:
                                capture = True
                                continue
                            elif line.strip() == "```" and capture:
                                break
                            elif capture:
                                table_lines.append(line)
                        
                        if table_lines:
                            st.markdown('<div class="table-container">', unsafe_allow_html=True)
                            st.markdown("\n".join(table_lines))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add copy button for the table
                            if st.button(f"📋 Copy Table (Chart {i+1})", key=f"copy_btn_{i}"):
                                st.code("\n".join(table_lines), language="markdown")
                        else:
                            st.info("No table found for this chart")
                    else:
                        st.info("No extraction found for this chart")
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No charts were detected in the document.")
    
    # Show the full markdown output in a collapsible section
    with st.expander("Full Extracted Markdown"):
        st.code(markdown_content, language="markdown")
    
    # Button to go back to upload
    if st.button("🔄 Process Another Document", use_container_width=True):
        st.session_state.active_tab = "upload"
        st.session_state.extracted_data = None
        st.session_state.charts_found = []
        st.rerun()

def main():
    """Main app function"""
    # Add a sidebar with information, including About section and file upload
    with st.sidebar:
        st.title("Graphscribe")
        st.markdown("---")
        
        # Add file upload to sidebar
        st.markdown("<h3>Upload Document</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload PDF or image containing charts", 
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supported formats: PDF, PNG, JPG, JPEG",
            on_change=clear_submit
        )
        
        # Update session state with uploaded file
        if uploaded_file is not None:
            # Check if it's a new file
            if st.session_state.uploaded_file is None or st.session_state.uploaded_file.name != uploaded_file.name:
                st.session_state.uploaded_file = uploaded_file
                # Save the file to a temporary location
                file_path = save_uploaded_file(uploaded_file)
                # Reset to upload tab if new file
                st.session_state.active_tab = "upload"
                st.session_state.extracted_data = None
                st.session_state.charts_found = []
        else:
            st.session_state.uploaded_file = None
        
        # Add model selection dropdown (superficial)
        st.markdown("---")
        st.markdown("<h3>Model Settings</h3>", unsafe_allow_html=True)
        
        model_options = {
            "Gemini Flash 2.5": "Google's fastest multimodal model for chart understanding",
            "Claude 3.7 Sonnet": "Anthropic's advanced vision model with chart extraction",
            "GPT-4o": "OpenAI's most capable multimodal model",
            "Qwen 2.5 VL": "Alibaba's vision-language model for chart analysis"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0,
            format_func=lambda x: x
        )
        
        # Display model description
        st.markdown(f"<p style='font-size: 0.85em; color: #64748b;'>{model_options[selected_model]}</p>", unsafe_allow_html=True)
        
        # Add Agent registration button
        st.button("Register as Agent", use_container_width=True)
            
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-content">
        <h3>How It Works</h3>
        
        <ol>
            <li><strong>Upload</strong> a PDF or image</li>
            <li><strong>Preview</strong> the document</li>
            <li><strong>Process</strong> to extract tables</li>
            <li><strong>Compare</strong> original charts with extracted tables</li>
            <li><strong>Download</strong> the results as Markdown</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-content">
        <h3>About Graphscribe</h3>
        <p>Graphscribe is an intelligent document understanding system that extracts structured data from charts, graphs, and diagrams in PDFs and images. It transforms visual information into clean, structured Markdown tables.</p>
        
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Process PDFs and images with charts/diagrams</li>
            <li>Use advanced AI for chart-to-table conversion</li>
            <li>Smart chart detection and table extraction</li>
            <li>ReAct-based agent architecture</li>
            <li>Export results in markdown format</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("© 2025 Graphscribe by Pawan Kumar")
    
    # Create tabs using a custom switcher to maintain state
    if st.session_state.active_tab == "upload":
        upload_interface()
    elif st.session_state.active_tab == "results":
        results_tab()

if __name__ == "__main__":
    main() 