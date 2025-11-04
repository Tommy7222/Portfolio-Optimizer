import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- 2. TITLE AND HEADER ---
st.title("Modern Portfolio Theory Optimizer 📈")
st.write("This tool helps you optimize a portfolio or analyze your own custom portfolio.")

# --- 3. HELPER FUNCTIONS ---

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
def get_display_fundamentals(tickers):
    """Fetches key fundamental data for a list of tickers for a display table."""
    fundamental_data = []
    for ticker_str in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_str)
            info = ticker_obj.info
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
        except Exception as e:
            st.warning(f"Could not fetch fundamentals for {ticker_str}: {e}")
            
    if not fundamental_data:
        return pd.DataFrame()
        
    return pd.DataFrame(fundamental_data).set_index("Ticker")

@st.cache_data
def get_numerical_fundamentals(tickers, key):
    """Fetches a specific numerical fundamental (e.g., 'forwardPE', 'pegRatio') for optimization."""
    metrics = {}
    for ticker_str in tickers:
        try:
            info = yf.Ticker(ticker_str).info
            metric = info.get(key)
            if metric is not None and isinstance(metric, (int, float)):
                if key == 'dividendYield':
                    metrics[ticker_str] = metric 
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
            info = yf.Ticker(t).info
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

# --- 4. APP MODE SELECTION ---
app_mode = st.radio(
    "What do you want to do?",
    ("Find an Optimal Portfolio", "Analyze My Custom Portfolio"),
    horizontal=True,
    label_visibility="collapsed"
)


# --- 5. SIDEBAR FOR USER INPUTS ---
with st.sidebar:
    st.header("Your Inputs")
    
    tickers_string = st.text_input(
        "Enter Tickers (comma-separated)", 
        "AAPL,MSFT,GOOG,AMZN,TSLA"
    )
    tickers = [s.strip().upper() for s in tickers_string.split(",") if s.strip()]
    
    time_horizon = st.selectbox(
        "Select Time Horizon",
        ("1y", "2y", "3y", "5y", "10y"),
        index=3
    )
    
    custom_inputs = {}
    
    if app_mode == "Find an Optimal Portfolio":
        st.subheader("Optimization Goal")
        optimization_choice = st.radio(
            "Choose Your Goal:",
            ("Best (Maximum Sharpe Ratio)", 
             "Best Value (Max Earnings Yield)", 
             "Growth at a Reasonable Price (Lowest PEG)", 
             "Safest (Minimum Volatility)"),
            key="goal"
        )
    
    else: # Analyze My Custom Portfolio
        st.subheader("Your Custom Portfolio")
        
        input_type = st.radio(
            "Input By:",
            ("Percentage Weights", "Number of Shares")
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
                    # --- FIX: Allow fractional shares up to 2 decimal places ---
                    custom_inputs[t] = st.number_input(f"Shares {t}", min_value=0.0, value=1.0, step=0.01, format="%.2f")
        else:
            st.info("Enter tickers above to set weights or shares.")
        
        optimization_choice = "" # Not used in this mode

    run_button = st.button("Run Analysis")


# --- 6. MAIN APP LOGIC ---
if run_button and tickers:
    
    if len(tickers) < 2:
        st.warning("Please enter at least two valid stock tickers.")
    else:
        # Create tabs
        if app_mode == "Find an Optimal Portfolio":
            tab1, tab2, tab3 = st.tabs(["📊 MPT Optimization", "📈 Correlation", "🏢 Fundamentals"])
        else:
            tab1, tab2, tab3 = st.tabs(["📝 Portfolio Report Card", "📈 Correlation", "🏢 Fundamentals"])

        try:
            # --- A. Fetch Price Data (Common to both modes) ---
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
                    # Format columns
                    for col in ["Trailing P/E", "Forward P/E", "PEG Ratio", "P/B Ratio"]:
                        fundamental_df[col] = pd.to_numeric(fundamental_df[col], errors='coerce').map('{:,.2f}'.format, na_action='ignore')
                    
                    div_yield_numeric = pd.to_numeric(fundamental_df["Div. Yield"], errors='coerce')
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
                        
                        elif optimization_choice == "Growth at a Reasonable Price (Lowest PEG)":
                            st.header("Growth at a Reasonable Price (Lowest PEG)")
                            st.write("This model uses 'PEG Yield' (1 / PEG Ratio) as the 'Expected Return'.")
                            metrics = get_numerical_fundamentals(valid_tickers, "pegRatio")
                            if not metrics:
                                st.error("Could not fetch any valid PEG Ratios for this optimization.")
                                st.stop()
                            mu_series = pd.Series({ticker: 1/peg for ticker, peg in metrics.items()})

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
                        elif optimization_choice == "Growth at a Reasonable Price (Lowest PEG)":
                            st.metric("Portfolio 'PEG Yield' Score", f"{performance[0]:.2f}")
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
                        
                        dollar_values = {t: custom_inputs[t] * prices.get(t, 0) for t in custom_inputs}
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

                    # Weighted PEG
                    pegs = get_numerical_fundamentals(valid_tickers, "pegRatio")
                    if pegs:
                        valid_peg_tickers = set(weights_dict.keys()) & set(pegs.keys())
                        weighted_peg = sum(weights_dict[ticker] * pegs[ticker] for ticker in valid_peg_tickers if pegs.get(ticker, 0) > 0)
                        col2.metric("Weighted PEG Ratio", f"{weighted_peg:.2f}")
                    else:
                        col2.metric("Weighted PEG Ratio", "N/A")

                    # Weighted Div. Yield
                    div_yields = get_numerical_fundamentals(valid_tickers, "dividendYield")
                    if div_yields:
                        valid_div_tickers = set(weights_dict.keys()) & set(div_yields.keys())
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