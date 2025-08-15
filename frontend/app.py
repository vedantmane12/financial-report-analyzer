import streamlit as st
import requests
import json
from datetime import datetime
import time
import pandas as pd
import re
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your API URL

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        background-color: #e7f3ff;
        color: #0366d6;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_available_options():
    """Get available options from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/options", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_sample_queries():
    """Get sample queries from API"""
    # Removed - no longer using sample queries
    return []

def test_individual_tool(tool_name, query, org):
    """Test individual tool"""
    # Removed - no longer using tool testing
    return {"success": False, "error": "Tool testing removed"}

def run_analysis(query, year=None, quarter=None, org=None, tools=None):
    """Run financial analysis via API"""
    try:
        payload = {
            "query": query,
            "year": year,
            "quarter": quarter,
            "org": org,
            "tools": tools or ["vector_search", "web_search", "yfinance_analysis"]
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with st.spinner("🤖 Running AI analysis... This may take up to 30 seconds."):
            response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=60)
        
        return response.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The analysis is taking longer than expected."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def parse_key_metrics(metrics_text):
    """Parse key metrics text into a structured table format"""
    if not metrics_text or metrics_text == "No key metrics identified":
        return None
    
    # Try to extract bullet points or structured data
    lines = metrics_text.split('\n')
    metrics_data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Key financial') or line.startswith('Key Metrics'):
            continue
            
        # Remove bullet points and clean up
        line = re.sub(r'^[•\-\*]\s*', '', line)
        
        # Try to extract metric name and value
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                metric_name = parts[0].strip()
                metric_value = parts[1].strip()
                metrics_data.append({
                    'Metric': metric_name,
                    'Value': metric_value
                })
        elif line.strip():
            # Handle cases where there's no colon separator
            metrics_data.append({
                'Metric': 'Financial Metric',
                'Value': line.strip()
            })
    
    if metrics_data:
        return pd.DataFrame(metrics_data)
    return None

def clean_financial_text(text):
    """Clean up corrupted financial text with pipe delimiters and formatting issues"""
    if not text:
        return text
    
    # Remove excessive pipe characters
    text = re.sub(r'\|+', ' ', text)
    
    # Simple string replacements to fix common issues
    text = text.replace('billionin', 'billion in')
    text = text.replace('millionin', 'million in')
    text = text.replace('%$', '%')
    
    # Fix basic spacing around numbers and percentages
    text = re.sub(r'up(\d+)%', r'up \1%', text)
    text = re.sub(r'down(\d+)%', r'down \1%', text)
    text = re.sub(r'(\d+)billion', r'\1 billion', text)
    text = re.sub(r'(\d+)million', r'\1 million', text)
    
    # Fix common concatenated words
    text = text.replace('Operating', ' Operating')
    text = text.replace('Market', ' Market')
    text = text.replace('Data', ' Data')
    text = text.replace('Platform', ' Platform')
    text = text.replace('Highlights', ' Highlights')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_financial_analysis(analysis_text):
    """Parse financial analysis text into structured sections and extract tabular data"""
    if not analysis_text or analysis_text == "No financial analysis available":
        return None, None
    
    # Split into paragraphs for better display
    paragraphs = [p.strip() for p in analysis_text.split('\n\n') if p.strip()]
    
    # Try to extract any tabular data from the text
    tables_data = []
    clean_paragraphs = []
    
    for para in paragraphs:
        processed = False
        
        # Look for profitability data patterns with parentheses and percentages
        if ('Income before income tax' in para or 'Net income' in para or 
            re.search(r'\(\d+\.\d+\)', para) and '%' in para):
            
            # This looks like profitability/margins data
            profit_data = []
            
            # Extract income tax data
            if 'Income before income tax' in para:
                tax_match = re.search(r'Income before income tax\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', para)
                if tax_match:
                    profit_data.append({
                        'Financial Metric': 'Income before income tax',
                        'Q1': f"${tax_match.group(1)}B",
                        'Q2': f"${tax_match.group(2)}B", 
                        'Q3': f"${tax_match.group(3)}B",
                        'Q4': f"${tax_match.group(4)}B"
                    })
            
            # Extract net income data
            if 'Net income' in para:
                # Look for pattern like: Net income | 51.0% | 11.5% | 45.0% | 14.2% |
                net_match = re.search(r'Net income\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%', para)
                if net_match:
                    profit_data.append({
                        'Financial Metric': 'Net Income Margin',
                        'Q1': f"{net_match.group(1)}%",
                        'Q2': f"{net_match.group(2)}%",
                        'Q3': f"{net_match.group(3)}%", 
                        'Q4': f"{net_match.group(4)}%"
                    })
            
            # Extract income tax expense data
            if 'Income tax expense' in para:
                tax_exp_match = re.search(r'Income tax expense.*?\|\s*([\d.]+)\s*\|\s*\(([\d.]+)\)\s*\|\s*([\d.]+)\s*\|\s*\(([\d.]+)\)', para)
                if tax_exp_match:
                    profit_data.append({
                        'Financial Metric': 'Income Tax Expense (Benefit)',
                        'Q1': f"${tax_exp_match.group(1)}B",
                        'Q2': f"(${tax_exp_match.group(2)}B)",
                        'Q3': f"${tax_exp_match.group(3)}B",
                        'Q4': f"(${tax_exp_match.group(4)}B)"
                    })
            
            if profit_data:
                tables_data.extend(profit_data)
                processed = True
        
        # Look for segment data patterns (like Data Center | 14, 514 | 3,833 | 29, 121...)
        elif ('|' in para and 
              any(segment in para for segment in ['Data Center', 'Gaming', 'Professional', 'Automotive', 'OEM', 'Total revenue']) and
              # Exclude simple text that might have | characters but isn't tabular data
              len(para.split('|')) > 3):
            
            # This looks like segment revenue data
            segments = para.split('|')
            segment_data = []
            
            i = 0
            while i < len(segments) - 1:
                segment_name = segments[i].strip()
                if segment_name and not segment_name.replace(',', '').replace('.', '').isdigit():
                    # Look for the next few values that might be revenue figures
                    values = []
                    j = i + 1
                    while j < len(segments) and j < i + 4:  # Take next 3 values max
                        value = segments[j].strip()
                        if value and (value.replace(',', '').replace('.', '').isdigit() or 
                                     any(char.isdigit() for char in value)):
                            values.append(value)
                        j += 1
                    
                    if values:
                        segment_data.append({
                            'Business Segment': segment_name,
                            'Current Period': values[0] if len(values) > 0 else 'N/A',
                            'Previous Period': values[1] if len(values) > 1 else 'N/A',
                            'Growth/Change': values[2] if len(values) > 2 else 'N/A'
                        })
                    i = j
                else:
                    i += 1
            
            if segment_data:
                tables_data.extend(segment_data)
                processed = True
        
        # Check if paragraph contains financial figures but isn't complex tabular data
        elif (re.search(r'\$[\d,]+\.?\d*\s*(billion|million|B|M)', para, re.IGNORECASE) and
              not re.search(r'[|]{3,}', para)):  # Avoid paragraphs with multiple | characters
            
            # Extract financial figures for simple cases only
            revenue_match = re.search(r'revenue.*?\$?([\d,]+\.?\d*)\s*(billion|million|B|M)', para, re.IGNORECASE)
            growth_match = re.search(r'(\d+\.?\d*)%.*?(growth|increase|decrease|up|down)', para, re.IGNORECASE)
            margin_match = re.search(r'margin.*?(\d+\.?\d*)%', para, re.IGNORECASE)
            
            if revenue_match or growth_match or margin_match:
                row_data = {'Category': 'Financial Metric'}
                if revenue_match:
                    row_data['Revenue'] = f"${revenue_match.group(1)} {revenue_match.group(2)}"
                if growth_match:
                    row_data['Growth Rate'] = f"{growth_match.group(1)}%"
                if margin_match:
                    row_data['Margin'] = f"{margin_match.group(1)}%"
                
                if len(row_data) > 1:  # More than just 'Category'
                    tables_data.append(row_data)
        
        # Only add to clean_paragraphs if not processed as table data
        if not processed:
            # Clean the text before adding to paragraphs
            cleaned_para = clean_financial_text(para)
            clean_paragraphs.append(cleaned_para)
    
    # Create DataFrame if we found structured data
    df = None
    if tables_data:
        df = pd.DataFrame(tables_data)
    
    return df, clean_paragraphs

def display_image_from_text(text):
    """Extract and display images referenced in text"""
    # Look for image references in the text
    image_pattern = r'!\s*([^\s]+\.(png|jpg|jpeg|gif))'
    matches = re.findall(image_pattern, text, re.IGNORECASE)
    
    for match in matches:
        image_filename = match[0]
        try:
            # Display image with proper caption
            st.image(image_filename, caption=f"Financial Chart: {image_filename}", use_column_width=True)
        except:
            st.warning(f"Could not display image: {image_filename}")
            # Show as a link instead
            st.markdown(f"📊 **Chart Reference**: {image_filename}")

def display_report(report_data):
    """Display the financial report in a single page format with proper formatting"""
    if not report_data.get("report"):
        st.error("No report data available")
        return
    
    report = report_data["report"]
    
    # Executive Summary first
    st.markdown("## 📋 Executive Summary")
    st.markdown(f'<div class="metric-card">{report["summary"]}</div>', unsafe_allow_html=True)
    
    # Key Metrics with proper table formatting
    st.markdown("## 📊 Key Metrics")
    metrics_df = parse_key_metrics(report["key_metrics"])
    
    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(
                    "Metric",
                    help="Financial metric name",
                    width="medium"
                ),
                "Value": st.column_config.TextColumn(
                    "Value",
                    help="Metric value and details",
                    width="large"
                )
            }
        )
    else:
        # Fallback to formatted text if parsing fails
        st.markdown("### Key Financial Metrics")
        st.markdown(report["key_metrics"])
    
    # Financial Analysis with structured display
    st.markdown("## 💰 Financial Analysis")
    analysis_df, analysis_paragraphs = parse_financial_analysis(report["financial_analysis"])
    
    # Display any extracted tables first
    if analysis_df is not None and not analysis_df.empty:
        st.markdown("### Financial Performance Summary")
        
        # Check if this is segment data, profitability data, or general financial data
        if 'Business Segment' in analysis_df.columns:
            # This is segment revenue data
            st.dataframe(
                analysis_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Business Segment": st.column_config.TextColumn("Business Segment", width="medium"),
                    "Current Period": st.column_config.TextColumn("Current Period ($ Millions)", width="medium"),
                    "Previous Period": st.column_config.TextColumn("Previous Period ($ Millions)", width="medium"),
                    "Growth/Change": st.column_config.TextColumn("Growth/Change", width="medium")
                }
            )
        elif 'Financial Metric' in analysis_df.columns and any(col.startswith('Q') for col in analysis_df.columns):
            # This is quarterly profitability data
            st.markdown("#### Quarterly Financial Metrics")
            st.dataframe(
                analysis_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Financial Metric": st.column_config.TextColumn("Metric", width="large"),
                    "Q1": st.column_config.TextColumn("Q1", width="small"),
                    "Q2": st.column_config.TextColumn("Q2", width="small"),
                    "Q3": st.column_config.TextColumn("Q3", width="small"),
                    "Q4": st.column_config.TextColumn("Q4", width="small")
                }
            )
        else:
            # General financial metrics
            st.dataframe(
                analysis_df,
                use_container_width=True,
                hide_index=True
            )
        st.markdown("---")
    
    # Display analysis text in organized sections
    if analysis_paragraphs:
        for i, paragraph in enumerate(analysis_paragraphs):
            # Check for images in the paragraph
            if re.search(r'!\s*[^\s]+\.(png|jpg|jpeg|gif)', paragraph, re.IGNORECASE):
                display_image_from_text(paragraph)
                # Remove image references from text
                paragraph = re.sub(r'!\s*[^\s]+\.(png|jpg|jpeg|gif)', '', paragraph, flags=re.IGNORECASE).strip()
            
            if paragraph:
                # Add section headers for better organization
                if i == 0:
                    st.markdown("### Revenue and Performance Analysis")
                elif i == 1:
                    st.markdown("### Profitability and Margins")
                elif i == 2:
                    st.markdown("### Market Position and Strategy")
                
                st.markdown(paragraph)
                
                if i < len(analysis_paragraphs) - 1:
                    st.markdown("---")
    else:
        st.markdown(report["financial_analysis"])
    
    # Real-Time Market Data
    st.markdown("## 📈 Real-Time Market Data")
    market_data = report["real_time_data"]
    
    # Try to extract market data into a table format
    if "MARKET DATA SUMMARY" in market_data or "REAL-TIME METRICS" in market_data:
        # Parse market data for tabular display
        lines = market_data.split('\n')
        market_metrics = []
        current_company = None
        
        for line in lines:
            line = line.strip()
            if line.endswith(':') and any(company in line for company in ['NVIDIA', 'AMD', 'INTEL']):
                current_company = line.replace(':', '')
            elif line.startswith('- ') and current_company:
                metric_text = line[2:].strip()
                if ':' in metric_text:
                    metric_name, metric_value = metric_text.split(':', 1)
                    market_metrics.append({
                        'Company': current_company,
                        'Metric': metric_name.strip(),
                        'Value': metric_value.strip()
                    })
        
        if market_metrics:
            st.markdown("### Real-Time Market Metrics")
            market_df = pd.DataFrame(market_metrics)
            st.dataframe(
                market_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Company": st.column_config.TextColumn("Company", width="small"),
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="large")
                }
            )
        else:
            st.markdown(market_data)
    else:
        st.markdown(market_data)
    
    # Market Insights
    st.markdown("## 🌐 Market Insights")
    insights_text = report["market_insights"]
    
    # Check for and display any referenced images
    if re.search(r'!\s*[^\s]+\.(png|jpg|jpeg|gif)', insights_text, re.IGNORECASE):
        display_image_from_text(insights_text)
        insights_text = re.sub(r'!\s*[^\s]+\.(png|jpg|jpeg|gif)', '', insights_text, flags=re.IGNORECASE).strip()
    
    st.markdown(insights_text)
    
    # Research Details
    st.markdown("## 🔍 Research Details")
    st.markdown("### Research Steps")
    st.markdown(report["research_steps"])
    st.markdown("### Sources")
    st.markdown(report["sources"])
    
    # Display metadata only once at the bottom
    if report_data.get("metadata"):
        st.markdown("---")
        with st.expander("📋 Analysis Metadata"):
            metadata = report_data["metadata"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Query Parameters:**")
                params = metadata.get("query_parameters", {})
                for key, value in params.items():
                    if value:
                        st.write(f"- {key.title()}: {value}")
            
            with col2:
                st.markdown("**Execution Stats:**")
                stats = metadata.get("execution_stats", {})
                for key, value in stats.items():
                    st.write(f"- {key.replace('_', ' ').title()}: {value}")

def main():
    # Header
    st.markdown('<div class="main-header">📊 Financial Report Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered financial analysis combining quarterly reports, real-time market data, and news insights</div>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not accessible. Please ensure the FastAPI backend is running on http://localhost:8000")
        st.info("To start the backend, run: `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    # Show API status
    with st.expander("🔧 API Status"):
        st.success("✅ API is running and healthy")
        if health_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("API Status", health_data["status"])
            with col2:
                st.metric("Services", len([s for s in health_data["services"].values() if s == "available"]))
            with col3:
                st.metric("Last Check", datetime.fromisoformat(health_data["timestamp"]).strftime("%H:%M:%S"))
    
    # Get available options
    options = get_available_options()
    if not options:
        st.warning("Could not load available options from API")
        return
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("🎛️ Analysis Parameters")
        
        # Year selection
        year = st.selectbox(
            "📅 Year:",
            options["years"],
            index=0,
            help="Select a specific fiscal year for analysis"
        )
        
        # Quarter selection
        quarter = st.multiselect(
            "📊 Quarter(s):",
            options["quarters"],
            help="Select specific quarters for analysis"
        )
        
        # Organization selection
        org = st.multiselect(
            "🏢 Organization(s):",
            options["organizations"],
            help="Select companies to analyze"
        )
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query_input = st.text_area(
            "💬 Enter your financial analysis question:",
            height=100,
            placeholder="e.g., Compare NVIDIA's Q3 2024 earnings with current stock performance",
            help="Ask any question about financial performance, market trends, or company comparisons"
        )
        
        # Analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not query_input.strip():
                st.error("❌ Please enter a query")
            else:
                # Run the analysis with all tools
                start_time = time.time()
                result = run_analysis(
                    query=query_input.strip(),
                    year=year,
                    quarter=quarter if quarter else None,
                    org=org if org else None,
                    tools=["vector_search", "web_search", "yfinance_analysis"]  # Always use all tools
                )
                execution_time = time.time() - start_time
                
                if result.get("success"):
                    st.markdown(f'<div class="success-message">✅ Analysis completed successfully in {execution_time:.1f} seconds!</div>', unsafe_allow_html=True)
                    
                    # Display the report (removed Tools Used section)
                    display_report(result)
                    
                else:
                    st.markdown(f'<div class="error-message">❌ Analysis failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.metric("Available Years", len(options["years"]))
        st.metric("Companies", len(options["organizations"]))
        st.metric("Analysis Tools", "3")
        
        # Recent activity placeholder
        st.markdown("### 🕒 System Info")
        st.info(f"API Status: Healthy\nLast Updated: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()