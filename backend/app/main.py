from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import sys
import os
from datetime import datetime
import logging

# Add the project root to Python path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from features.langgraph_ai import run_agents
except ImportError as e:
    print(f"Warning: Could not import langgraph_ai: {e}")
    run_agents = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Report Analyzer API",
    description="AI-powered financial analysis combining quarterly reports, real-time market data, and news insights",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    query: str = Field(..., description="The financial analysis question", min_length=10)
    year: Optional[str] = Field(None, description="Specific year (2023, 2024, 2025)")
    quarter: Optional[List[str]] = Field(None, description="List of quarters (Q1, Q2, Q3, Q4)")
    org: Optional[List[str]] = Field(None, description="List of organizations (NVIDIA, AMD, INTEL)")
    tools: Optional[List[str]] = Field(
        default=["vector_search", "web_search", "yfinance_analysis"],
        description="Tools to use for analysis"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Compare NVIDIA's Q3 2024 earnings with current stock performance",
                "year": "2024",
                "quarter": ["Q3"],
                "org": ["NVIDIA"],
                "tools": ["vector_search", "web_search", "yfinance_analysis"]
            }
        }

class FinancialReport(BaseModel):
    research_steps: str
    key_metrics: str
    financial_analysis: str
    real_time_data: str
    market_insights: str
    summary: str
    sources: str

class AnalysisResponse(BaseModel):
    success: bool
    report: Optional[FinancialReport] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    tools_used: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

class QuickTestRequest(BaseModel):
    tool_name: str = Field(..., description="Tool to test (vector_search, web_search, yfinance_analysis)")
    query: str = Field(default="test query", description="Test query")
    org: Optional[List[str]] = Field(default=["NVIDIA"], description="Organizations to test")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API status and dependencies"""
    services_status = {
        "langgraph_ai": "available" if run_agents else "unavailable",
        "api": "running",
        "timestamp": datetime.now().isoformat()
    }
    
    return HealthResponse(
        status="healthy" if run_agents else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services_status
    )

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_financial_data(request: AnalysisRequest):
    """
    Main endpoint for financial analysis using AI agents
    
    Combines multiple data sources:
    - Historical quarterly reports (Pinecone)
    - Real-time market data (YFinance) 
    - Current news and insights (Web Search)
    """
    
    if not run_agents:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: langgraph_ai module not loaded"
        )
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting analysis for query: {request.query}")
        logger.info(f"Parameters - Year: {request.year}, Quarter: {request.quarter}, Org: {request.org}")
        
        # Validate year if provided
        if request.year and request.year not in ["2023", "2024", "2025"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid year: {request.year}. Must be 2023, 2024, or 2025"
            )
        
        # Validate organizations if provided
        valid_orgs = ["NVIDIA", "AMD", "INTEL"]
        if request.org:
            invalid_orgs = [org for org in request.org if org not in valid_orgs]
            if invalid_orgs:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid organizations: {invalid_orgs}. Must be from {valid_orgs}"
                )
        
        # Validate quarters if provided
        valid_quarters = ["Q1", "Q2", "Q3", "Q4"]
        if request.quarter:
            invalid_quarters = [q for q in request.quarter if q not in valid_quarters]
            if invalid_quarters:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid quarters: {invalid_quarters}. Must be from {valid_quarters}"
                )
        
        # Run the AI agents
        runnable, initial_state = run_agents(
            query=request.query,
            tool_keys=request.tools,
            year=request.year,
            quarter=request.quarter,
            org=request.org
        )
        
        # Execute with appropriate recursion limit
        config = {"recursion_limit": 20}
        result = runnable.invoke(initial_state, config=config)
        
        # Extract results
        tools_used = []
        final_report = None
        
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if step.tool != "final_answer":
                    tools_used.append(step.tool)
                elif step.tool == "final_answer" and step.log != "TBD":
                    try:
                        # Parse the final report
                        if isinstance(step.log, str):
                            report_data = json.loads(step.log)
                        else:
                            report_data = step.log
                        
                        final_report = FinancialReport(**report_data)
                        break
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error parsing final report: {e}")
                        # Fallback: treat as plain text
                        final_report = FinancialReport(
                            research_steps="Analysis completed",
                            key_metrics="Unable to parse metrics",
                            financial_analysis=str(step.log)[:1000] + "..." if len(str(step.log)) > 1000 else str(step.log),
                            real_time_data="Data extracted",
                            market_insights="Analysis performed",
                            summary="Report generated successfully",
                            sources="Multiple data sources"
                        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare metadata
        metadata = {
            "query_parameters": {
                "year": request.year,
                "quarter": request.quarter,
                "org": request.org,
                "tools_requested": request.tools
            },
            "execution_stats": {
                "total_steps": len(result.get("intermediate_steps", [])),
                "unique_tools_used": len(set(tools_used)),
                "execution_time_seconds": execution_time
            }
        }
        
        if final_report:
            logger.info(f"Analysis completed successfully in {execution_time:.2f} seconds")
            return AnalysisResponse(
                success=True,
                report=final_report,
                execution_time=execution_time,
                tools_used=list(set(tools_used)),
                metadata=metadata
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Analysis completed but no final report was generated"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}",
            execution_time=execution_time
        )

# Quick tool testing endpoint
@app.post("/test-tool")
async def test_individual_tool(request: QuickTestRequest):
    """Test individual tools for debugging and monitoring"""
    
    if not run_agents:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: langgraph_ai module not loaded"
        )
    
    try:
        # Import individual tools
        from features.langgraph_ai import web_search, vector_search, yfinance_analysis
        
        tool_map = {
            "web_search": web_search,
            "vector_search": vector_search,
            "yfinance_analysis": yfinance_analysis
        }
        
        if request.tool_name not in tool_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool: {request.tool_name}. Available: {list(tool_map.keys())}"
            )
        
        tool_func = tool_map[request.tool_name]
        
        # Prepare tool arguments based on tool type
        if request.tool_name == "web_search":
            result = tool_func.invoke({"query": request.query})
        elif request.tool_name == "vector_search":
            result = tool_func.invoke({
                "query": request.query,
                "year": "2024",
                "quarter": ["Q3"],
                "org": request.org
            })
        elif request.tool_name == "yfinance_analysis":
            result = tool_func.invoke({
                "query": request.query,
                "org": request.org,
                "year": "2024",
                "quarter": ["Q3"]
            })
        
        return {
            "success": True,
            "tool": request.tool_name,
            "query": request.query,
            "result": result[:500] + "..." if len(str(result)) > 500 else result,
            "result_length": len(str(result))
        }
    
    except Exception as e:
        return {
            "success": False,
            "tool": request.tool_name,
            "error": str(e)
        }

# Get available options
@app.get("/options")
async def get_available_options():
    """Get available years, quarters, organizations, and tools"""
    return {
        "years": ["2023", "2024", "2025"],
        "quarters": ["Q1", "Q2", "Q3", "Q4"],
        "organizations": ["NVIDIA", "AMD", "INTEL"],
        "tools": ["vector_search", "web_search", "yfinance_analysis"],
        "tool_descriptions": {
            "vector_search": "Historical quarterly reports from Pinecone database",
            "web_search": "Current news and market insights",
            "yfinance_analysis": "Real-time stock data and financial metrics"
        }
    }

# Sample queries endpoint
@app.get("/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing and demonstration"""
    return {
        "sample_queries": [
            {
                "title": "Single Company Analysis",
                "query": "Analyze NVIDIA's Q3 2024 financial performance and current market position",
                "year": "2024",
                "quarter": ["Q3"],
                "org": ["NVIDIA"]
            },
            {
                "title": "Multi-Company Comparison",
                "query": "Compare NVIDIA, AMD, and Intel Q3 2024 earnings with current stock performance",
                "year": "2024",
                "quarter": ["Q3"],
                "org": ["NVIDIA", "AMD", "INTEL"]
            },
            {
                "title": "Real-time Market Analysis",
                "query": "What are the current market valuations and P/E ratios for semiconductor companies?",
                "org": ["NVIDIA", "AMD", "INTEL"]
            },
            {
                "title": "Historical Trend Analysis",
                "query": "Show revenue growth trends for NVIDIA from 2023 to 2024",
                "year": "2024",
                "quarter": ["Q1", "Q2", "Q3", "Q4"],
                "org": ["NVIDIA"]
            },
            {
                "title": "Industry Overview",
                "query": "What are the latest trends in the semiconductor industry and market outlook?",
                "year": "2024",
                "org": ["NVIDIA", "AMD", "INTEL"]
            }
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Financial Report Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/analyze": "Main analysis endpoint",
            "/health": "Health check",
            "/test-tool": "Test individual tools",
            "/options": "Get available parameters",
            "/sample-queries": "Get sample queries",
            "/docs": "API documentation"
        },
        "description": "AI-powered financial analysis combining quarterly reports, real-time market data, and news insights"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)