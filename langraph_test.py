import json
from features.langgraph_ai import run_agents

def format_report(report_str):
    """Format the final report for better readability"""
    try:
        # Try to parse as JSON if it's a string
        if isinstance(report_str, str):
            report = json.loads(report_str)
        else:
            report = report_str
            
        print("\n" + "="*80)
        print("# FINANCIAL RESEARCH REPORT")
        print("="*80)
        
        print("\n## RESEARCH STEPS:")
        print("-" * 40)
        print(report.get('research_steps', 'N/A'))
        
        print("\n## KEY METRICS:")
        print("-" * 40)
        print(report.get('key_metrics', 'N/A'))
        
        print("\n## FINANCIAL ANALYSIS:")
        print("-" * 40)
        print(report.get('financial_analysis', 'N/A'))
        
        print("\n## REAL-TIME MARKET DATA:")
        print("-" * 40)
        print(report.get('real_time_data', 'N/A'))
        
        print("\n## MARKET INSIGHTS:")
        print("-" * 40)
        print(report.get('market_insights', 'N/A'))
        
        print("\n## EXECUTIVE SUMMARY:")
        print("-" * 40)
        print(report.get('summary', 'N/A'))
        
        print("\n## SOURCES:")
        print("-" * 40)
        print(report.get('sources', 'N/A'))
        
        print("\n" + "="*80)
        
    except json.JSONDecodeError:
        print("Error parsing report. Raw output:")
        print(report_str)
    except Exception as e:
        print(f"Error formatting report: {e}")
        print("Raw output:", report_str)

def main():
    # Test queries - now with YFinance integration examples
    queries = [
        {
            "query": "Compare NVIDIA's Q3 2024 reported earnings with current stock performance and market valuation",
            "year": "2024",
            "quarter": ["Q3"],
            "org": ["NVIDIA"],
            "tools": ["vector_search", "web_search", "yfinance_analysis"]
        },
        {
            "query": "Analyze AMD and Intel's 2023 financial performance vs their stock price movements",
            "year": "2023",
            "quarter": ["Q1", "Q2", "Q3", "Q4"],
            "org": ["AMD", "INTEL"],
            "tools": ["vector_search", "yfinance_analysis"]
        },
        {
            "query": "What are the latest semiconductor industry trends with real-time market data?",
            "year": "2024",
            "quarter": ["Q4"],
            "org": ["NVIDIA", "AMD", "INTEL"],
            "tools": ["vector_search", "web_search", "yfinance_analysis"]
        },
        {
            "query": "Show NVIDIA's current market metrics and P/E ratio compared to Q2 2024 earnings",
            "year": "2024",
            "quarter": ["Q2"],
            "org": ["NVIDIA"],
            "tools": ["vector_search", "yfinance_analysis"]
        }
    ]
    
    # Select which query to run (change index to test different queries)
    test_query = queries[0]  # Change this index to test different queries
    
    print(f"\n### Query: {test_query['query']}")
    print(f"### Year: {test_query.get('year', 'All')}")
    print(f"### Quarter(s): {test_query.get('quarter', 'All')}")
    print(f"### Organization(s): {test_query.get('org', 'All')}")
    print(f"### Tools: {test_query.get('tools', ['vector_search', 'web_search', 'yfinance_analysis'])}")
    print("\nStarting research...\n")
    
    # Run the agent with specified tools
    runnable, initial_state = run_agents(
        query=test_query['query'],
        tool_keys=test_query.get('tools', ["vector_search", "web_search", "yfinance_analysis"]),
        year=test_query.get('year'),
        quarter=test_query.get('quarter'),
        org=test_query.get('org')
    )
    
    # Execute the graph with increased recursion limit
    config = {"recursion_limit": 20}  # Increased from 10 to allow more tool calls
    result = runnable.invoke(initial_state, config=config)
    
    # Extract and format the final report
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if step.tool == "final_answer" and step.log != "TBD":
                format_report(step.log)
                break
    else:
        print("No results found")
    
    # Show if charts were generated
    if result.get("charts"):
        print("\n#### Visualizations generated (stored in state)")

def test_individual_tools():
    """Test individual tools to ensure they work correctly"""
    from features.langgraph_ai import web_search, vector_search, yfinance_analysis
    
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL TOOLS")
    print("="*50)
    
    # Test web search
    print("\n1. Testing Web Search...")
    try:
        web_result = web_search.invoke({"query": "NVIDIA earnings Q4 2024"})
        if web_result and "Error" not in web_result:
            print("#### Web search working")
            print(f"   Preview: {web_result[:200]}...")
        else:
            print("#### Web search failed:", web_result)
    except Exception as e:
        print("#### Web search error:", e)
    
    # Test vector search
    print("\n2. Testing Vector Search...")
    try:
        vector_result = vector_search.invoke({
            "query": "revenue growth",
            "year": "2024",
            "quarter": ["Q1"],
            "org": ["NVIDIA"]
        })
        if vector_result and "No relevant" not in vector_result:
            print("#### Vector search working")
            print(f"   Found {vector_result.count('Chunk')} chunks")
        else:
            print("#### Vector search failed:", vector_result)
    except Exception as e:
        print("#### Vector search error:", e)
    
    # Test YFinance
    print("\n3. Testing YFinance Analysis...")
    try:
        yfinance_result = yfinance_analysis.invoke({
            "query": "stock performance",
            "year": "2024",
            "quarter": ["Q3"],
            "org": ["NVIDIA"],
            "analysis_type": "comprehensive"
        })
        if yfinance_result and "Error" not in yfinance_result:
            print("#### YFinance analysis working")
            if "MARKET DATA SUMMARY" in yfinance_result:
                print("#### Market data retrieved")
            if "REAL-TIME METRICS" in yfinance_result:
                print("#### Real-time metrics calculated")
            print(f"   Preview: {yfinance_result[:200]}...")
        else:
            print("#### YFinance analysis failed:", yfinance_result)
    except Exception as e:
        print("#### YFinance error:", e)
    
    print("\n" + "="*50)

def test_comparison_analysis():
    """Test multi-company comparison with all tools"""
    print("\n" + "="*60)
    print("MULTI-COMPANY COMPARISON TEST")
    print("="*60)
    
    query = "Compare NVIDIA, AMD, and Intel Q3 2024 earnings with their current stock prices and market caps"
    
    print(f"\n Query: {query}")
    print("\nRunning comprehensive analysis with all tools...\n")
    
    # Run with all three tools
    runnable, initial_state = run_agents(
        query=query,
        tool_keys=["vector_search", "web_search", "yfinance_analysis"],
        year="2024",
        quarter=["Q3"],
        org=["NVIDIA", "AMD", "INTEL"]
    )
    
    # Execute with increased recursion limit
    config = {"recursion_limit": 20}  # Increased to allow more thorough analysis
    result = runnable.invoke(initial_state, config=config)
    
    # Display results
    if "intermediate_steps" in result:
        tools_used = []
        for step in result["intermediate_steps"]:
            if step.tool != "final_answer":
                tools_used.append(step.tool)
            elif step.tool == "final_answer" and step.log != "TBD":
                print(f"\nTools used: {', '.join(set(tools_used))}")
                format_report(step.log)
                break

def test_yfinance_only():
    """Test using only YFinance for real-time data"""
    print("\n" + "="*60)
    print("YFINANCE REAL-TIME DATA TEST")
    print("="*60)
    
    query = "Get current market data and stock metrics for NVIDIA"
    
    print(f"\n” Query: {query}")
    print("\nFetching real-time market data...\n")
    
    # Run with only YFinance - don't pass None values
    runnable, initial_state = run_agents(
        query=query,
        tool_keys=["yfinance_analysis"],
        org=["NVIDIA"]
        # Don't pass year and quarter if they're None
    )
    
    # Execute with increased recursion limit for YFinance
    config = {"recursion_limit": 15}  # Increased from 10
    result = runnable.invoke(initial_state, config=config)
    
    # Display results
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if step.tool == "yfinance_analysis":
                print("YFinance Data Retrieved:")
                print(step.log)
            elif step.tool == "final_answer" and step.log != "TBD":
                format_report(step.log)
                break

if __name__ == "__main__":
    import sys
    
    # Command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-tools":
            test_individual_tools()
        elif sys.argv[1] == "--test-comparison":
            test_comparison_analysis()
        elif sys.argv[1] == "--test-yfinance":
            test_yfinance_only()
        elif sys.argv[1] == "--all":
            test_individual_tools()
            print("\n" + "="*60 + "\n")
            main()
            print("\n" + "="*60 + "\n")
            test_comparison_analysis()
        else:
            print("Usage: python langraph_test.py [--test-tools|--test-comparison|--test-yfinance|--all]")
    else:
        # Run default test
        main()
        
        print("\n" + "-"*60)
        print("TIP: Run with different options:")
        print("  python langraph_test.py --test-tools      # Test individual tools")
        print("  python langraph_test.py --test-comparison  # Test multi-company comparison")
        print("  python langraph_test.py --test-yfinance   # Test YFinance only")
        print("  python langraph_test.py --all             # Run all tests")