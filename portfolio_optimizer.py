import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- 2. TITLE AND HEADER ---
st.title("Modern Portfolio Theory Optimizer 📈")
st.write("A tool for portfolio optimization, custom analysis, and single-stock deep dives.")

# --- 3. SESSION STATE ---
# Initialize session state to remember if analysis has been run
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

def reset_analysis():
    """Callback to reset analysis if inputs change."""
    st.session_state.run_analysis = False

# --- 4. HELPER FUNCTIONS ---

@st.cache_data
def get_raw_stock_data(tickers, period):
    """Downloads raw historical stock data, auto-adjusted for splits/dividends."""
    return yf.download(tickers, period=period, auto_adjust=True)

@st.cache_data
def get_market_data(period, market_ticker="SPY"):
    """Downloads market data (e.g., SPY) for beta calculation, auto-adjusted."""
    return yf.download(market_ticker, period=period, auto_adjust=True)

@st.cache_data
def get_rf_rate():
    """Fetches the 10-Yr Treasury Yield as the risk-free rate."""
    try:
        rf_data = yf.Ticker("^TNX").history(period="7d")
        if not rf_data.empty:
            return rf_data["Close"].iloc[-1] / 100
    except Exception:
        pass
    return 0.02

@st.cache_data
def get_stock_info(ticker_str):
    """Fetches the .info dictionary for a single stock."""
    try:
        return yf.Ticker(ticker_str).info
    except Exception as e:
        st.error(f"Error fetching info for {ticker_str}: {e}")
        return None

@st.cache_data
def get_historical_financials(ticker_str):
    """Fetches historical annual AND quarterly financials and cash flow."""
    try:
        t = yf.Ticker(ticker_str)
        annual_financials = t.financials
        annual_cashflow = t.cashflow
        quarterly_financials = t.quarterly_financials
        quarterly_cashflow = t.quarterly_cashflow
        return annual_financials, annual_cashflow, quarterly_financials, quarterly_cashflow
    except Exception as e:
        st.error(f"Error fetching financials for {ticker_str}: {e}")
        return None, None, None, None

@st.cache_data
def get_display_fundamentals(tickers):
    """Fetches key fundamental data for a list of tickers for a display table."""
    fundamental_data = []
    for ticker_str in tickers:
        info = get_stock_info(ticker_str)
        if info:
            data = {
                "Ticker": ticker_str,
                "Company Name": info.get("shortName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Trailing P/E": info.get("trailingPE", "N/A"),
                "Forward P/E": info.get("forwardPE", "N/A"),
                "PEG Ratio": info.get("pegRatio", "N/A"),
                "P/B Ratio": info.get("priceToBook", "N/A"),
                "Div. Yield": info.get("dividendYield", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
            }
            fundamental_data.append(data)
            
    if not fundamental_data:
        return pd.DataFrame()
        
    return pd.DataFrame(fundamental_data).set_index("Ticker")

@st.cache_data
def get_numerical_fundamentals(tickers, key):
    """Fetches a specific numerical fundamental (e.g., 'forwardPE', 'priceToBook') for optimization."""
    metrics = {}
    for ticker_str in tickers:
        try:
            info = get_stock_info(ticker_str)
            metric = info.get(key)
            if metric is not None and isinstance(metric, (int, float)):
                if key == 'dividendYield':
                    # --- FIX: Assume API returns percentage (e.g., 0.73), divide by 100 ---
                    metrics[ticker_str] = metric / 100.0 
                elif metric > 0: 
                    metrics[ticker_str] = metric
        except Exception:
            pass
    return metrics

@st.cache_data
def get_latest_prices(tickers):
    """Fetches the latest price for a list of tickers."""
    prices = {}
    for t in tickers:
        try:
            info = get_stock_info(t)
            price = info.get('currentPrice', info.get('previousClose'))
            if price:
                prices[t] = price
            else:
                st.warning(f"Could not get current price for {t}")
        except Exception as e:
            st.error(f"Error fetching price for {t}: {e}")
    return prices

def calculate_portfolio_beta(portfolio_returns, market_returns):
    """Calculates the beta of a portfolio given its returns and market returns."""
    covariance = portfolio_returns.cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance
    return beta

def get_diversification_grade(corr_matrix):
    """Calculates an average pairwise correlation and assigns a grade."""
    lower_triangle = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool))
    avg_corr = lower_triangle.stack().mean()
    
    if np.isnan(avg_corr):
        return "N/A", "Could not calculate correlation grade.", 0.0

    if avg_corr < 0.3:
        grade = "A+"
        note = "Excellent Diversification. Stocks are highly uncorrelated."
    elif avg_corr < 0.4:
        grade = "A"
        note = "Great Diversification. Stocks have very low correlation."
    elif avg_corr < 0.5:
        grade = "B"
        note = "Good Diversification. Stocks have low correlation."
    elif avg_corr < 0.6:
        grade = "C"
        note = "Average Diversification. Consider adding less correlated assets."
    elif avg_corr < 0.7:
        grade = "D"
        note = "Poor Diversification. Stocks are moving together."
    else:
        grade = "F"
        note = "Very Poor Diversification. Portfolio has high unsystematic risk."
        
    return grade, note, avg_corr

def format_large_number(num):
    """Formats large numbers (e.g., 1.5B, 1.2T) for display."""
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    if abs(num) > 1e12:
        return f"{num / 1e12:.2f} T"
    if abs(num) > 1e9:
        return f"{num / 1e9:.2f} B"
    if abs(num) > 1e6:
        return f"{num / 1e6:.2f} M"
    if abs(num) > 1e3:
        return f"{num / 1e3:.2f} K"
    return f"{num:.2f}"

# --- 5. APP MODE SELECTION ---
app_mode = st.radio(
    "What do you want to do?",
    ("Find an Optimal Portfolio", "Analyze My Custom Portfolio", "Single Stock Deep Dive"),
    horizontal=True,
    label_visibility="collapsed",
    key="app_mode_selector",
    on_change=reset_analysis # Reset if mode changes
)


# --- 6. SIDEBAR FOR USER INPUTS ---
with st.sidebar:
    st.header("Your Inputs")
    
    if app_mode == "Single Stock Deep Dive":
        tickers_string = st.text_input(
            "Enter a Ticker (e.g., AAPL)", 
            "AAPL",
            on_change=reset_analysis # Reset if ticker changes
        )
    else:
        tickers_string = st.text_input(
            "Enter Tickers (comma-separated)", 
            "AAPL,MSFT,GOOG,AMZN,TSLA",
            on_change=reset_analysis # Reset if tickers change
        )
    
    tickers = [s.strip().upper() for s in tickers_string.split(",") if s.strip()]
    
    time_horizon = st.selectbox(
        "Select Time Horizon",
        ("1y", "2y", "3y", "5y", "10y"),
        index=3,
        on_change=reset_analysis # Reset if horizon changes
    )
    
    custom_inputs = {}
    
    if app_mode == "Find an Optimal Portfolio":
        st.subheader("Optimization Goal")
        optimization_choice = st.radio(
            "Choose Your Goal:",
            ("Best (Maximum Sharpe Ratio)", 
             "Best Value (Max Earnings Yield)", 
             "Deep Value (Lowest P/B Ratio)", # <-- CHANGED FROM PEG
             "Safest (Minimum Volatility)"),
            key="goal",
            on_change=reset_analysis # Reset if goal changes
        )
    
    elif app_mode == "Analyze My Custom Portfolio":
        st.subheader("Your Custom Portfolio")
        
        input_type = st.radio(
            "Input By:",
            ("Percentage Weights", "Number of Shares"),
            on_change=reset_analysis # Reset if input type changes
        )
        
        st.write("---")
        
        if tickers:
            if input_type == "Percentage Weights":
                st.write("Enter % (e.g., 25 for 25%):")
                for t in tickers:
                    custom_inputs[t] = st.number_input(f"Weight % {t}", min_value=0.0, max_value=100.0, value=100.0/len(tickers), step=1.0)
            
            else: # Number of Shares
                st.write("Enter Number of Shares:")
                for t in tickers:
                    custom_inputs[t] = st.number_input(f"Shares {t}", min_value=0.0, value=1.0, step=0.01, format="%.2f")
        else:
            st.info("Enter tickers above to set weights or shares.")
        
        optimization_choice = "" # Not used in this mode
    
    else: # Single Stock Deep Dive
        optimization_choice = ""
        # No other inputs needed

    # When this button is clicked, set the state to True
    if st.button("Run Analysis"):
        st.session_state.run_analysis = True


# --- 7. MAIN APP LOGIC ---
# Check if the button was clicked OR if it was already clicked and inputs haven't changed
if (st.session_state.run_analysis) and tickers:
    
    # =========================================================
    # MODE 3: SINGLE STOCK DEEP DIVE
    # =========================================================
    if app_mode == "Single Stock Deep Dive":
        
        if len(tickers) > 1:
            st.info(f"Analyzing only the first ticker: {tickers[0]}. To analyze multiple, use another mode.")
        
        ticker = tickers[0]
        
        try:
            info = get_stock_info(ticker)
            if not info or info.get('quoteType') == 'MUTUALFUND': 
                st.error(f"Could not retrieve valid data for {ticker}. It may be an unsupported asset type (like a mutual fund) or an invalid ticker.")
                st.stop()
            
            # --- Stock Header ---
            st.header(f"{info.get('shortName', ticker)} ({ticker})")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Current Price", f"${info.get('currentPrice', info.get('previousClose', 'N/A')):.2f}")
            c2.metric("Market Cap", format_large_number(info.get('marketCap')))
            
            trailing_pe = info.get('trailingPE')
            c3.metric("P/E Ratio", f"{trailing_pe:.2f}" if isinstance(trailing_pe, (int, float)) else "N/A")
            
            forward_pe = info.get('forwardPE')
            c4.metric("Forward P/E", f"{forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else "N/A")

            beta = info.get('beta')
            c5.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A")
            
            st.caption(f"{info.get('sector', 'N/A')} | {info.get('industry', 'N/A')} | {info.get('website', '')}")
            
            # --- Create Tabs ---
            tab_chart, tab_financials, tab_profile = st.tabs(["📈 Price Chart", "📊 Financials", "🏢 Company Profile"])
            
            with tab_profile:
                st.subheader("Company Profile")
                st.write(info.get('longBusinessSummary', 'No summary available.'))
            
            with tab_chart:
                st.subheader(f"Price History ({time_horizon})")
                price_data = get_raw_stock_data([ticker], time_horizon)
                if not price_data.empty:
                    y_data = price_data['Close'].squeeze()
                    fig_price = px.line(x=price_data.index, y=y_data, title=f"{ticker} Adjusted Close Price")
                    fig_price.update_layout(xaxis_title="Date", yaxis_title="Price ($)")
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.warning("Could not load price chart data.")
            
            with tab_financials:
                st.subheader("Historical Financials")
                
                # --- Annual vs Quarterly Toggle ---
                period_type = st.radio(
                    "Select Period:",
                    ("Annual", "Quarterly"),
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                annual_fin, annual_cf, qtr_fin, qtr_cf = get_historical_financials(ticker)
                
                fin_df, cf_df = None, None
                
                if period_type == "Annual":
                    if annual_fin is not None and not annual_fin.empty:
                        fin_df = annual_fin.transpose().iloc[::-1]
                    if annual_cf is not None and not annual_cf.empty:
                        cf_df = annual_cf.transpose().iloc[::-1]
                else: # Quarterly
                    if qtr_fin is not None and not qtr_fin.empty:
                        fin_df = qtr_fin.transpose().iloc[::-1]
                    if qtr_cf is not None and not qtr_cf.empty:
                        cf_df = qtr_cf.transpose().iloc[::-1]

                
                if fin_df is not None and cf_df is not None:
                    # --- Prepare Data ---
                    fin_df.index = pd.to_datetime(fin_df.index)
                    cf_df.index = pd.to_datetime(cf_df.index)

                    fin_df = fin_df.head(4)
                    cf_df = cf_df.head(4)
                    
                    if period_type == "Annual":
                        fin_df.index = fin_df.index.strftime('%Y')
                        cf_df.index = cf_df.index.strftime('%Y')
                    else: # Quarterly
                        fin_df.index = fin_df.index.strftime('%Y-%m')
                        cf_df.index = cf_df.index.strftime('%Y-%m')

                    # --- Create formatted text columns (B/M/T) ---
                    if "Total Revenue" in fin_df:
                        fin_df['Total Revenue Text'] = fin_df['Total Revenue'].apply(format_large_number)
                    if "Net Income" in fin_df:
                        fin_df['Net Income Text'] = fin_df['Net Income'].apply(format_large_number)
                    if "Operating Cash Flow" in cf_df:
                        cf_df['Operating Cash Flow Text'] = cf_df['Operating Cash Flow'].apply(format_large_number)
                    if "Operating Cash Flow" in cf_df and "Capital Expenditure" in cf_df:
                        cf_df["Free Cash Flow"] = cf_df["Operating Cash Flow"].fillna(0) - cf_df["Capital Expenditure"].fillna(0)
                        cf_df['Free Cash Flow Text'] = cf_df['Free Cash Flow'].apply(format_large_number)


                    # --- Create Charts ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "Total Revenue" in fin_df:
                            fig_rev = px.line(fin_df, x=fin_df.index, y="Total Revenue", title=f"Total Revenue ({period_type})", text='Total Revenue Text', markers=True, line_shape='spline')
                            fig_rev.update_traces(texttemplate='%{text}', textposition='top center')
                            fig_rev.update_layout(xaxis_title="Period", yaxis_title="Amount ($)")
                            st.plotly_chart(fig_rev, use_container_width=True)
                        else:
                            st.warning("Revenue data not available.")

                        if "Operating Cash Flow" in cf_df:
                            fig_ocn = px.line(cf_df, x=cf_df.index, y="Operating Cash Flow", title=f"Operating Cash Flow ({period_type})", text='Operating Cash Flow Text', markers=True, line_shape='spline')
                            fig_ocn.update_traces(texttemplate='%{text}', textposition='top center')
                            fig_ocn.update_layout(xaxis_title="Period", yaxis_title="Amount ($)")
                            st.plotly_chart(fig_ocn, use_container_width=True)
                        else:
                          st.warning("Operating Cash Flow data not available.")

                    with col2:
                        if "Net Income" in fin_df:
                            fig_ni = px.line(fin_df, x=fin_df.index, y="Net Income", title=f"Net Income ({period_type})", text='Net Income Text', markers=True, line_shape='spline')
                            fig_ni.update_traces(texttemplate='%{text}', textposition='top center')
                            fig_ni.update_layout(xaxis_title="Period", yaxis_title="Amount ($)")
                            st.plotly_chart(fig_ni, use_container_width=True)
                        else:
                            st.warning("Net Income data not available.")
                        
                        if "Free Cash Flow" in cf_df:
                            fig_fcf = px.line(cf_df, x=cf_df.index, y="Free Cash Flow", title=f"Free Cash Flow ({period_type})", text='Free Cash Flow Text', markers=True, line_shape='spline')
                            fig_fcf.update_traces(texttemplate='%{text}', textposition='top center')
                            fig_fcf.update_layout(xaxis_title="Period", yaxis_title="Amount ($)")
                            st.plotly_chart(fig_fcf, use_container_width=True)
                        else:
                            st.warning("Free Cash Flow data not available (missing Operating Cash Flow or Capital Expenditure).")
                
                else:
                    st.error(f"Could not retrieve {period_type.lower()} financial statements. This data may not be available for this asset.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e) # Print full traceback for debugging

    # =========================================================
    # MODES 1 & 2: PORTFOLIO ANALYSIS
    # =========================================================
    elif len(tickers) < 2:
        st.warning("Please enter at least two valid stock tickers for portfolio analysis.")
    else:
        # Create tabs
        if app_mode == "Find an Optimal Portfolio":
            tab1, tab2, tab3 = st.tabs(["📊 MPT Optimization", "📈 Correlation", "🏢 Fundamentals"])
        else:
            tab1, tab2, tab3 = st.tabs(["📝 Portfolio Report Card", "📈 Correlation", "🏢 Fundamentals"])

        try:
            # --- A. Fetch Price Data ---
            raw_data = get_raw_stock_data(tickers, time_horizon)
            
            if raw_data.empty:
                st.error("Could not fetch price data. Are the tickers correct or is there a network issue?")
                st.stop()
            
            # --- B. Select 'Close' prices (which are auto-adjusted) ---
            if len(tickers) == 1:
                data = raw_data[['Close']]
                data.columns = tickers 
            else:
                data = raw_data['Close']
            
            if data.empty or data.isnull().all().all():
                st.error("No valid 'Close' price data found after download. Check tickers and time horizon.")
                st.stop()
            
            # --- C. Handle missing data ---
            data = data.dropna(axis=1, how='all').dropna(axis=0, how='any')
            
            valid_tickers = data.columns.tolist()
            if len(data.columns) < 2:
                st.error(f"Not enough valid data after cleaning. Need at least 2 tickers. Found: {', '.join(data.columns)}")
                st.stop()
            
            if len(valid_tickers) < len(tickers):
                st.info(f"Analysis running on: {', '.join(valid_tickers)} (dropped tickers with missing price data).")
                custom_inputs = {t: v for t, v in custom_inputs.items() if t in valid_tickers}
                if not custom_inputs and app_mode == "Analyze My Custom Portfolio":
                    st.error("None of your entered tickers have valid price data.")
                    st.stop()

            # --- D. Get Risk-Free Rate & Risk Model (Common) ---
            rf_rate = get_rf_rate()
            S = risk_models.sample_cov(data)

            # --- E. Correlation Tab (Common) ---
            with tab2:
                st.subheader("Stock Correlation Matrix")
                st.write("This heatmap shows how closely your selected stocks move together. (1 = perfectly correlated, -1 = perfectly opposite).")
                corr_matrix = data.pct_change().corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                     title="Stock Correlation Heatmap",
                                     color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)

            # --- F. Fundamentals Tab (Common) ---
            with tab3:
                st.subheader("Fundamental Analysis")
                st.write("Use this data to cross-reference the MPT optimization with company fundamentals.")
                fundamental_df = get_display_fundamentals(tickers)
                if not fundamental_df.empty:
                    for col in ["Trailing P/E", "Forward P/E", "PEG Ratio", "P/B Ratio"]:
                        fundamental_df[col] = pd.to_numeric(fundamental_df[col], errors='coerce').map('{:,.2f}'.format, na_action='ignore')
                    
                    # --- FIX: Assume API returns percentage (e.g., 0.73) and divide by 100 ---
                    div_yield_numeric = pd.to_numeric(fundamental_df["Div. Yield"], errors='coerce') / 100.0
                    fundamental_df["Div. Yield"] = div_yield_numeric.map('{:,.2%}'.format, na_action='ignore')

                    fundamental_df["Market Cap"] = pd.to_numeric(fundamental_df["Market Cap"], errors='coerce').map('{:,.0f}'.format, na_action='ignore')
                    st.dataframe(fundamental_df)
                else:
                    st.info("No fundamental data could be fetched.")


            # --- G. Run Mode-Specific Logic ---
            with tab1:
                
                # =========================================================
                # MODE 1: FIND AN OPTIMAL PORTFOLIO
                # =========================================================
                if app_mode == "Find an Optimal Portfolio":
                    
                    if optimization_choice == "Best (Maximum Sharpe Ratio)":
                        st.header("Best (Maximum Sharpe Ratio) Portfolio")
                        st.write("This model uses *historical returns* to find the portfolio with the best risk-adjusted return.")
                        mu = expected_returns.mean_historical_return(data)
                        ef = EfficientFrontier(mu, S)
                        weights = ef.max_sharpe(risk_free_rate=rf_rate)
                    
                    elif optimization_choice == "Safest (Minimum Volatility)":
                        st.header("Safest (Minimum Volatility) Portfolio")
                        st.write("This model ignores returns and finds the portfolio with the *lowest volatility* (risk).")
                        mu = expected_returns.mean_historical_return(data) 
                        ef = EfficientFrontier(mu, S)
                        weights = ef.min_volatility()
                    
                    else: # Value or Growth Models
                        if optimization_choice == "Best Value (Max Earnings Yield)":
                            st.header("Best Value (Max Earnings Yield) Portfolio")
                            st.write("This model uses the 'Forward Earnings Yield' (1 / Forward P/E) as the 'Expected Return'.")
                            metrics = get_numerical_fundamentals(valid_tickers, "forwardPE")
                            if not metrics:
                                st.error("Could not fetch any valid Forward P/E ratios for this optimization.")
                                st.stop()
                            mu_series = pd.Series({ticker: 1/pe for ticker, pe in metrics.items()})
                        
                        elif optimization_choice == "Deep Value (Lowest P/B Ratio)": # <-- CHANGED
                            st.header("Deep Value (Lowest P/B Ratio)") # <-- CHANGED
                            st.write("This model uses 'Book Yield' (1 / P/B Ratio) as the 'Expected Return'.") # <-- CHANGED
                            metrics = get_numerical_fundamentals(valid_tickers, "priceToBook") # <-- CHANGED
                            if not metrics:
                                st.error("Could not fetch any valid P/B Ratios for this optimization.") # <-- CHANGED
                                st.stop()
                            mu_series = pd.Series({ticker: 1/pb for ticker, pb in metrics.items()}) # <-- CHANGED

                        common_tickers = list(set(S.columns) & set(mu_series.index))
                        if len(common_tickers) < 2:
                            st.error(f"Not enough common data (Price History + Fundamentals) to optimize. Need at least 2 tickers.")
                            st.stop()
                        
                        mu = mu_series[common_tickers]
                        S_filtered = S.loc[common_tickers, common_tickers]
                        st.info(f"Optimizing on {len(common_tickers)} stocks with valid data: {', '.join(common_tickers)}")
                        
                        ef = EfficientFrontier(mu, S_filtered)
                        weights = ef.max_sharpe(risk_free_rate=0.0) 

                    
                    cleaned_weights = ef.clean_weights()
                    performance = ef.portfolio_performance(verbose=False, risk_free_rate=rf_rate)
                    
                    st.subheader("Optimal Portfolio Weights")
                    col1, col2 = st.columns(2)
                    with col1:
                        weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
                        weights_df = weights_df[weights_df['Weight'] > 0]
                        weights_df.reset_index(inplace=True)
                        weights_df.rename(columns={'index': 'Ticker'}, inplace=True)
                        
                        fig_pie = px.pie(weights_df, names='Ticker', values='Weight', title='Optimal Portfolio Allocation', hover_data=['Ticker'])
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        st.subheader("Key Performance Stats")
                        if optimization_choice == "Best Value (Max Earnings Yield)":
                            st.metric("Portfolio Earnings Yield", f"{performance[0]:.2%}")
                        elif optimization_choice == "Deep Value (Lowest P/B Ratio)": # <-- CHANGED
                            st.metric("Portfolio 'Book Yield' Score", f"{performance[0]:.2f}") # <-- CHANGED
                        else:
                            st.metric("Expected Annual Return", f"{performance[0]:.2%}")
                        
                        st.metric("Annual Volatility (Risk)", f"{performance[1]:.2%}")
                        st.metric("Sharpe Ratio", f"{performance[2]:.2f}")
                        st.dataframe(pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight']))

                # =========================================================
                # MODE 2: ANALYZE MY CUSTOM PORTFOLIO
                # =========================================================
                elif app_mode == "Analyze My Custom Portfolio":
                    st.header("Portfolio Report Card")
                    
                    weights_dict = {}
                    
                    # --- 1. Calculate Weights ---
                    if input_type == "Percentage Weights":
                        total_pct = sum(custom_inputs.values())
                        if not np.isclose(total_pct, 100.0) and total_pct > 0:
                            st.warning(f"Weights sum to {total_pct:.2f}%, not 100%. Normalizing weights.")
                            weights_dict = {t: w / total_pct for t, w in custom_inputs.items()}
                        elif total_pct == 0:
                            st.error("Weights sum to zero. Please enter valid percentages.")
                            st.stop()
                        else:
                            weights_dict = {t: w / 100.0 for t, w in custom_inputs.items()}
                    
                    else: # Number of Shares
                        prices = get_latest_prices(valid_tickers)
                        if not prices:
                            st.error("Could not fetch current prices to calculate weights from shares.")
                            st.stop()
                        
                        dollar_values = {t: custom_inputs.get(t, 0) * prices.get(t, 0) for t in valid_tickers if t in custom_inputs}
                        total_portfolio_value = sum(dollar_values.values())
                        
                        if total_portfolio_value == 0:
                            st.error("Portfolio value is zero. Could not fetch prices or all share counts are zero.")
                            st.stop()
                        
                        weights_dict = {t: v / total_portfolio_value for t, v in dollar_values.items()}

                    
                    weights_array = np.array([weights_dict.get(t, 0) for t in valid_tickers])
                    
                    st.subheader("Your Custom Weights")
                    col1, col2 = st.columns(2)
                    with col1:
                        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
                        weights_df = weights_df[weights_df['Weight'] > 0]
                        weights_df.reset_index(inplace=True)
                        weights_df.rename(columns={'index': 'Ticker'}, inplace=True)
                        
                        fig_pie = px.pie(weights_df, names='Ticker', values='Weight', title='Your Custom Allocation', hover_data=['Ticker'])
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # --- 2. Calculate Performance Metrics ---
                    with col2:
                        st.subheader("Performance (Historical)")
                        mu_hist = expected_returns.mean_historical_return(data)
                        
                        ef = EfficientFrontier(mu_hist, S)
                        ef.set_weights(weights_dict) 
                        perf = ef.portfolio_performance(risk_free_rate=rf_rate)
                        
                        st.metric("Expected Annual Return", f"{perf[0]:.2%}")
                        st.metric("Annual Volatility (Risk)", f"{perf[1]:.2%}")
                        st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                        
                        if input_type == "Number of Shares":
                            st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")

                    st.divider()
                    
                    # --- 3. Risk & Diversification Grade ---
                    st.subheader("Risk & Diversification Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Beta
                        market_data = get_market_data(time_horizon)
                        market_returns = market_data['Close'].pct_change().squeeze() 
                        
                        portfolio_returns = (data.pct_change() * weights_array).sum(axis=1).squeeze()
                        
                        combined_returns = pd.DataFrame({
                            'portfolio': portfolio_returns,
                            'market': market_returns
                        }).dropna() 
                        
                        if not combined_returns.empty:
                            beta = calculate_portfolio_beta(combined_returns['portfolio'], combined_returns['market'])
                            st.metric("Portfolio Beta vs. SPY", f"{beta:.2f}")
                        else:
                            st.metric("Portfolio Beta vs. SPY", "N/A")
                            st.warning("Could not calculate Beta (no overlapping market/portfolio data).")
                        
                    with col2:
                        # Diversification Grade
                        grade, note, avg_corr = get_diversification_grade(corr_matrix)
                        st.metric(f"Diversification Grade: {grade}", f"Avg. Correlation: {avg_corr:.3f}")
                        st.caption(note)

                    st.divider()

                    # --- 4. Fundamental Profile ---
                    st.subheader("Fundamental Profile (Weighted)")
                    col1, col2, col3 = st.columns(3)
                    
                    # Weighted Fwd P/E
                    fwd_pes = get_numerical_fundamentals(valid_tickers, "forwardPE")
                    if fwd_pes:
                        valid_pe_tickers = set(weights_dict.keys()) & set(fwd_pes.keys())
                        weighted_pe = sum(weights_dict[ticker] * fwd_pes[ticker] for ticker in valid_pe_tickers if fwd_pes.get(ticker, 0) > 0)
                        col1.metric("Weighted Forward P/E", f"{weighted_pe:.2f}")
                    else:
                        col1.metric("Weighted Forward P/E", "N/A")

                    # Weighted P/B Ratio <-- CHANGED
                    pbs = get_numerical_fundamentals(valid_tickers, "priceToBook")
                    if pbs:
                        valid_pb_tickers = set(weights_dict.keys()) & set(pbs.keys())
                        weighted_pb = sum(weights_dict[ticker] * pbs[ticker] for ticker in valid_pb_tickers if pbs.get(ticker, 0) > 0)
                        col2.metric("Weighted P/B Ratio", f"{weighted_pb:.2f}")
                    else:
                        col2.metric("Weighted P/B Ratio", "N/A")

                    # Weighted Div. Yield
                    div_yields = get_numerical_fundamentals(valid_tickers, "dividendYield")
                    if div_yields:
                        valid_div_tickers = set(weights_dict.keys()) & set(div_yields.keys())
                        # --- FIX: div_yields is already / 100 from get_numerical_fundamentals ---
                        weighted_div = sum(weights_dict[ticker] * div_yields[ticker] for ticker in valid_div_tickers)
                        col3.metric("Weighted Dividend Yield", f"{weighted_div:.2%}")
                    else:
                        col3.metric("Weighted Dividend Yield", "N/A")


        # --- 7. ERROR HANDLING ---
        except OptimizationError as e:
            st.error(f"Optimization failed. This can happen if all stocks are perfectly correlated or if there's not enough data. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e) # Print full traceback for debugging
            
# --- 8. INITIAL PAGE CONTENT ---
else:
    if not tickers:
        st.info("Enter your tickers in the sidebar to begin.")