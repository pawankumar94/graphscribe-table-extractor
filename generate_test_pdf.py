#!/usr/bin/env python3
"""
Generate test PDF with financial charts and tables for testing extraction.

This script creates a sample PDF with financial data including charts and tables
similar to the "Relationships Between CAP and Financial Performance" example.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from io import BytesIO
import tempfile

# Create directory for saving files
os.makedirs('markdown/samples', exist_ok=True)

def format_currency(x, pos):
    """Format y-axis ticks as currency."""
    return f"${x:,.0f}"

def format_with_commas(x, pos=None):
    """Format numbers with commas for thousands separator."""
    if x >= 1000:
        return f"{x:,.0f}"
    return str(int(x))

def create_cap_vs_tsr_chart():
    """Create a chart showing NEO CAP versus TSR similar to the example."""
    # Sample data
    years = ['Fiscal 2021', 'Fiscal 2022', 'Fiscal 2023', 'Fiscal 2024']
    neo_cap = [79.6, 105.5, -64.1, 234.1]
    other_neos = [27.9, 38.5, -61.4, 85.6]
    nvidia_tsr = [141.39, 158.12, 133.89, 190.57]
    nasdaq_tsr = [207.79, 365.66, 326.34, 978.42]
    
    # Create figure with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Set position for bars
    x = np.arange(len(years))
    width = 0.2
    
    # Plot bars
    neo_bars = ax1.bar(x - width*1.5, neo_cap, width, label='NEO CAP', color='green', alpha=0.7)
    other_bars = ax1.bar(x - width/2, other_neos, width, label='Other NEOs Average CAP', color='lightgreen', alpha=0.7)
    
    # Plot lines on secondary axis
    nvidia_line = ax2.plot(x, nvidia_tsr, marker='o', linestyle='-', color='blue', label='NVIDIA TSR')
    nasdaq_line = ax2.plot(x, nasdaq_tsr, marker='s', linestyle='-', color='red', label='Nasdaq100 Index TSR')
    
    # Add value labels to bars
    for i, bars in enumerate([neo_bars, other_bars]):
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                valign = 'top'
                offset = -15
            else:
                valign = 'bottom'
                offset = 3
            ax1.annotate(f'${abs(height):,.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=valign,
                        fontsize=9)
    
    # Add axis labels and titles
    ax1.set_xlabel('')
    ax1.set_ylabel('CAP ($M)', fontsize=10)
    ax2.set_ylabel('TSR (%)', fontsize=10)
    plt.title('NEO CAP versus TSR', fontsize=12, fontweight='bold')
    
    # Set x-axis ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    
    # Add horizontal line at y=0
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add negative values at bottom of chart
    for i, v in enumerate(neo_cap):
        if v < 0:
            ax1.annotate(f'(${abs(v):,.1f})', xy=(i - width*1.5, -20), ha='center', fontsize=9, color='red')
    
    for i, v in enumerate(other_neos):
        if v < 0:
            ax1.annotate(f'(${abs(v):,.1f})', xy=(i - width/2, -20), ha='center', fontsize=9, color='red')
    
    # Add price markers for TSR line
    for i, v in enumerate(nvidia_tsr):
        ax2.annotate(f'${v:,.2f}', xy=(i, v), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom',
                    fontsize=8)
    
    for i, v in enumerate(nasdaq_tsr):
        ax2.annotate(f'${v:,.2f}', xy=(i, v), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom',
                    fontsize=8)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='NEO CAP'),
        Patch(facecolor='lightgreen', alpha=0.7, label='Other NEOs Average CAP'),
        Line2D([0], [0], color='blue', marker='o', label='NVIDIA TSR'),
        Line2D([0], [0], color='red', marker='s', label='Nasdaq100 Index TSR')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=4, frameon=False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to file for reference
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.imshow(plt.imread(BytesIO(img_data.getvalue())))
    plt.savefig('markdown/samples/cap_vs_tsr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return img_data.getvalue()

def create_cap_vs_income_chart():
    """Create a chart showing NEO CAP versus Net Income & Non-GAAP Operating Income."""
    # Sample data
    years = ['Fiscal 2021', 'Fiscal 2022', 'Fiscal 2023', 'Fiscal 2024']
    neo_cap = [79.6, 105.5, -64.1, 234.1]
    other_neos = [27.9, 38.5, -61.4, 85.6]
    net_income = [4332, 9752, 4368, 29760]
    non_gaap = [6803, 12690, 9040, 37134]
    
    # Create figure with secondary y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Set position for bars
    x = np.arange(len(years))
    width = 0.2
    
    # Plot bars
    neo_bars = ax1.bar(x - width*1.5, neo_cap, width, label='NEO CAP', color='green', alpha=0.7)
    other_bars = ax1.bar(x - width/2, other_neos, width, label='Other NEOs Average CAP', color='lightgreen', alpha=0.7)
    
    # Plot bars on secondary axis
    net_income_bars = ax2.bar(x + width/2, net_income, width, label='Net Income', color='grey', alpha=0.7)
    non_gaap_bars = ax2.bar(x + width*1.5, non_gaap, width, label='Non-GAAP Operating Income', color='darkgrey', alpha=0.7)
    
    # Add value labels to bars
    for i, bars in enumerate([neo_bars, other_bars]):
        for bar in bars:
            height = bar.get_height()
            if height < 0:
                valign = 'top'
                offset = -15
            else:
                valign = 'bottom'
                offset = 3
            ax1.annotate(f'${abs(height):,.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=valign,
                        fontsize=9)
    
    # Add value labels to secondary bars
    for bars in [net_income_bars, non_gaap_bars]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'${height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # Add axis labels and titles
    ax1.set_xlabel('')
    ax1.set_ylabel('CAP ($M)', fontsize=10)
    ax2.set_ylabel('Income ($M)', fontsize=10)
    plt.title('NEO CAP versus Net Income & Non-GAAP Operating Income', fontsize=12, fontweight='bold')
    
    # Set x-axis ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    
    # Add horizontal line at y=0
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add negative values at bottom of chart
    for i, v in enumerate(neo_cap):
        if v < 0:
            ax1.annotate(f'(${abs(v):,.1f})', xy=(i - width*1.5, -20), ha='center', fontsize=9, color='red')
    
    for i, v in enumerate(other_neos):
        if v < 0:
            ax1.annotate(f'(${abs(v):,.1f})', xy=(i - width/2, -20), ha='center', fontsize=9, color='red')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='NEO CAP'),
        Patch(facecolor='lightgreen', alpha=0.7, label='Other NEOs Average CAP'),
        Patch(facecolor='grey', alpha=0.7, label='Net Income'),
        Patch(facecolor='darkgrey', alpha=0.7, label='Non-GAAP Operating Income')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=2, frameon=False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save to file for reference
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.imshow(plt.imread(BytesIO(img_data.getvalue())))
    plt.savefig('markdown/samples/cap_vs_income.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return img_data.getvalue()

def create_sample_pdf():
    """Create a sample PDF with financial data charts and tables."""
    # Create document
    pdf_path = 'markdown/samples/financial_performance_test.pdf'
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a custom style for headers
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading2'],
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    # Create a list to hold the content
    content = []
    
    # Add title
    content.append(Paragraph("Relationships Between CAP and Financial Performance", title_style))
    content.append(Spacer(1, 12))
    
    # Add introductory text
    intro_text = """
    The following graphs illustrate how CAP for our NEOs aligns with the Company's financial performance measures as
    detailed in the Pay Versus Performance table above for each of Fiscal 2021, 2022, 2023, and 2024, as well as between the
    TSRs of NVIDIA and the Nasdaq100 Index, reflecting the value of a fixed $100 investment beginning with the market
    close on January 24, 2020, the last trading day before our fiscal 2021, through and including the end of the respective
    listed fiscal years.
    """
    content.append(Paragraph(intro_text, normal_style))
    content.append(Spacer(1, 12))
    
    # Generate and add first chart
    content.append(Paragraph("NEO CAP versus TSR", header_style))
    content.append(Spacer(1, 6))
    chart1_data = create_cap_vs_tsr_chart()
    img1 = Image(BytesIO(chart1_data), width=6*inch, height=3.5*inch)
    content.append(img1)
    content.append(Spacer(1, 12))
    
    # Add table for NEO CAP versus TSR
    content.append(Spacer(1, 6))
    cap_tsr_data = [
        ['Fiscal Year', 'CEO CAP', 'Other NEOs Average CAP', 'NVIDIA TSR', 'Nasdaq100 Index TSR'],
        ['Fiscal 2021', '$79.6', '$27.9', '$141.39', '$207.79'],
        ['Fiscal 2022', '$105.5', '$38.5', '$158.12', '$365.66'],
        ['Fiscal 2023', '($64.1)', '($61.4)', '$133.89', '$326.34'],
        ['Fiscal 2024', '$234.1', '$85.6', '$190.57', '$978.42']
    ]
    
    table1 = Table(cap_tsr_data, colWidths=[1.2*inch, 1.2*inch, 1.5*inch, 1.2*inch, 1.5*inch])
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(table1)
    
    content.append(Paragraph("*Note: Values on right y-axis range from ($20) to $1,120*", styles['Italic']))
    content.append(Spacer(1, 24))
    
    # Generate and add second chart
    content.append(Paragraph("NEO CAP versus Net Income & Non-GAAP Operating Income", header_style))
    content.append(Spacer(1, 6))
    chart2_data = create_cap_vs_income_chart()
    img2 = Image(BytesIO(chart2_data), width=6*inch, height=3.5*inch)
    content.append(img2)
    content.append(Spacer(1, 12))
    
    # Add table for NEO CAP versus Net Income & Non-GAAP Operating Income
    content.append(Spacer(1, 6))
    cap_income_data = [
        ['Fiscal Year', 'CEO CAP', 'Other NEOs Average CAP', 'Net Income', 'Non-GAAP Operating Income'],
        ['Fiscal 2021', '$79.6', '$27.9', '$4,332', '$6,803'],
        ['Fiscal 2022', '$105.5', '$38.5', '$9,752', '$12,690'],
        ['Fiscal 2023', '($64.1)', '($61.4)', '$4,368', '$9,040'],
        ['Fiscal 2024', '$234.1', '$85.6', '$29,760', '$37,134']
    ]
    
    table2 = Table(cap_income_data, colWidths=[1.2*inch, 1.2*inch, 1.5*inch, 1.2*inch, 1.7*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    content.append(table2)
    
    content.append(Paragraph("*Note: Values on right y-axis range from ($2,000) to $40,000*", styles['Italic']))
    content.append(Spacer(1, 24))
    
    # Add disclaimer
    disclaimer_text = """
    All information provided above under the "Pay Versus Performance" heading will not be deemed to be incorporated by reference into any filing
    of the Company under the Securities Act of 1933, as amended, or the Securities Exchange Act of 1934, as amended, whether made before or
    after the date hereof and irrespective of any general incorporation language in any such filing, except to the extent the Company specifically
    incorporates such information by reference.
    """
    content.append(Paragraph(disclaimer_text, styles['Italic']))
    
    # Build the PDF
    doc.build(content)
    
    print(f"Sample PDF created: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_sample_pdf() 