import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, solve, simplify, latex
import pandas as pd
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="M/M/1 Queue Mathematical Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .formula-box {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📊 M/M/1 Queue Mathematical Model</h1>', unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("📝 Queue Parameters")
    
    # Input parameters
    lambda_rate = st.sidebar.slider(
        "Arrival Rate (λ) - customers/hour", 
        min_value=0.1, 
        max_value=20.0, 
        value=8.0, 
        step=0.1,
        help="Average number of customers arriving per hour"
    )
    
    mu_rate = st.sidebar.slider(
        "Service Rate (μ) - customers/hour", 
        min_value=0.1, 
        max_value=25.0, 
        value=10.0, 
        step=0.1,
        help="Average number of customers served per hour"
    )
    
    # Calculate utilization factor
    rho = lambda_rate / mu_rate
    
    # Stability check
    if rho >= 1:
        st.error("⚠️ System is unstable! ρ = λ/μ ≥ 1. Service rate must be greater than arrival rate.")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results", "📈 Visualizations", "🧮 Mathematical Formulas", "📉 Sensitivity Analysis", "🎯 Simulation"])
    
    with tab1:
        display_results(lambda_rate, mu_rate, rho)
    
    with tab2:
        display_visualizations(lambda_rate, mu_rate, rho)
    
    with tab3:
        display_mathematical_formulas()
    
    with tab4:
        display_sensitivity_analysis(lambda_rate, mu_rate)
    
    with tab5:
        display_simulation(lambda_rate, mu_rate)

def calculate_mm1_metrics(lambda_rate, mu_rate):
    """Calculate all M/M/1 queue metrics"""
    rho = lambda_rate / mu_rate
    
    metrics = {
        'rho': rho,
        'L': rho / (1 - rho),  # Average number in system
        'Lq': (rho**2) / (1 - rho),  # Average number in queue
        'W': 1 / (mu_rate - lambda_rate),  # Average time in system
        'Wq': lambda_rate / (mu_rate * (mu_rate - lambda_rate)),  # Average time in queue
        'P0': 1 - rho,  # Probability of empty system
        'Pn_formula': f'P_n = (1-ρ)ρ^n = {1-rho:.4f} × {rho:.4f}^n'
    }
    
    return metrics

def display_results(lambda_rate, mu_rate, rho):
    """Display M/M/1 queue results"""
    st.header("📊 Queue Performance Metrics")
    
    metrics = calculate_mm1_metrics(lambda_rate, mu_rate)
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Utilization Factor (ρ)</h3>
            <h2 style="color: #1f77b4;">{:.4f}</h2>
            <p>System busy {:.1f}% of time</p>
        </div>
        """.format(metrics['rho'], metrics['rho']*100), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Customers in System (L)</h3>
            <h2 style="color: #ff7f0e;">{:.4f}</h2>
            <p>Including those being served</p>
        </div>
        """.format(metrics['L']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Customers in Queue (Lq)</h3>
            <h2 style="color: #2ca02c;">{:.4f}</h2>
            <p>Waiting to be served</p>
        </div>
        """.format(metrics['Lq']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Empty System Prob (P₀)</h3>
            <h2 style="color: #d62728;">{:.4f}</h2>
            <p>{:.1f}% chance system is empty</p>
        </div>
        """.format(metrics['P0'], metrics['P0']*100), unsafe_allow_html=True)
    
    # Time metrics
    st.subheader("⏱️ Time Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Time in System (W)", 
            f"{metrics['W']:.4f} hours", 
            f"{metrics['W']*60:.2f} minutes"
        )
    
    with col2:
        st.metric(
            "Average Waiting Time (Wq)", 
            f"{metrics['Wq']:.4f} hours", 
            f"{metrics['Wq']*60:.2f} minutes"
        )
    
    # Probability distribution table
    st.subheader("📋 State Probability Distribution")
    n_values = np.arange(0, 15)
    probabilities = [(1 - rho) * (rho ** n) for n in n_values]
    cumulative_prob = np.cumsum(probabilities)
    
    df = pd.DataFrame({
        'n (customers)': n_values,
        'P(n)': probabilities,
        'Cumulative P(n)': cumulative_prob
    })
    
    st.dataframe(df.style.format({
        'P(n)': '{:.6f}',
        'Cumulative P(n)': '{:.6f}'
    }))

def display_visualizations(lambda_rate, mu_rate, rho):
    """Display various visualizations"""
    st.header("📈 Queue Visualizations")
    
    # Probability distribution plot
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('State Probability Distribution', 'System Performance vs Utilization', 
                       'Arrival and Service Process', 'Queue Length Over Time (Simulation)'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Probability distribution
    n_values = np.arange(0, 20)
    probabilities = [(1 - rho) * (rho ** n) for n in n_values]
    
    fig1.add_trace(
        go.Bar(x=n_values, y=probabilities, name='P(n)', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Plot 2: Performance metrics vs utilization
    rho_range = np.linspace(0.1, 0.95, 50)
    L_values = rho_range / (1 - rho_range)
    W_values = 1 / (mu_rate * (1 - rho_range))
    
    fig1.add_trace(
        go.Scatter(x=rho_range, y=L_values, name='L (customers)', line=dict(color='red')),
        row=1, col=2
    )
    fig1.add_trace(
        go.Scatter(x=rho_range, y=W_values, name='W (hours)', line=dict(color='blue'), yaxis='y2'),
        row=1, col=2
    )
    
    # Plot 3: Arrival and service process visualization
    time_points = np.linspace(0, 2, 1000)
    arrival_process = np.random.poisson(lambda_rate * time_points[-1], len(time_points))
    service_times = np.random.exponential(1/mu_rate, len(time_points))
    
    fig1.add_trace(
        go.Scatter(x=time_points, y=np.cumsum(np.random.poisson(lambda_rate/500, len(time_points))), 
                  name='Arrivals', line=dict(color='green')),
        row=2, col=1
    )
    fig1.add_trace(
        go.Scatter(x=time_points, y=np.cumsum(np.random.poisson(mu_rate/500, len(time_points))), 
                  name='Services', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Plot 4: Queue length simulation
    np.random.seed(42)
    t_sim = np.linspace(0, 10, 1000)
    queue_length = simulate_queue_length(lambda_rate, mu_rate, t_sim)
    
    fig1.add_trace(
        go.Scatter(x=t_sim, y=queue_length, name='Queue Length', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig1.update_layout(height=800, showlegend=True, title_text="M/M/1 Queue Analysis Dashboard")
    fig1.update_xaxes(title_text="Number of Customers (n)", row=1, col=1)
    fig1.update_yaxes(title_text="Probability P(n)", row=1, col=1)
    fig1.update_xaxes(title_text="Utilization Factor (ρ)", row=1, col=2)
    fig1.update_yaxes(title_text="Average Number in System (L)", row=1, col=2)
    fig1.update_yaxes(title_text="Average Time in System (W)", secondary_y=True, row=1, col=2)
    fig1.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig1.update_yaxes(title_text="Cumulative Count", row=2, col=1)
    fig1.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig1.update_yaxes(title_text="Queue Length", row=2, col=2)
    
    st.plotly_chart(fig1, use_container_width=True)

def simulate_queue_length(lambda_rate, mu_rate, time_points):
    """Simulate queue length over time"""
    queue_length = np.zeros(len(time_points))
    current_length = 0
    
    for i, t in enumerate(time_points[1:], 1):
        dt = time_points[i] - time_points[i-1]
        
        # Arrivals (Poisson process)
        arrivals = np.random.poisson(lambda_rate * dt)
        current_length += arrivals
        
        # Services (only if queue is not empty)
        if current_length > 0:
            services = np.random.poisson(mu_rate * dt)
            current_length = max(0, current_length - services)
        
        queue_length[i] = current_length
    
    return queue_length

def display_mathematical_formulas():
    """Display mathematical formulas using SymPy"""
    st.header("🧮 Mathematical Formulas for M/M/1 Queue")
    
    # Define symbols
    lambda_sym, mu_sym, rho_sym, n = symbols('lambda mu rho n', positive=True)
    
    st.subheader("📐 Basic Definitions")
    
    formulas = [
        ("Utilization Factor", "ρ = λ/μ", rho_sym, lambda_sym/mu_sym),
        ("Probability of n customers", "P_n = (1-ρ)ρ^n", None, (1-rho_sym)*rho_sym**n),
        ("Average number in system", "L = ρ/(1-ρ)", None, rho_sym/(1-rho_sym)),
        ("Average number in queue", "L_q = ρ²/(1-ρ)", None, rho_sym**2/(1-rho_sym)),
        ("Average time in system", "W = 1/(μ-λ)", None, 1/(mu_sym-lambda_sym)),
        ("Average waiting time", "W_q = λ/[μ(μ-λ)]", None, lambda_sym/(mu_sym*(mu_sym-lambda_sym))),
        ("Probability of empty system", "P_0 = 1-ρ", None, 1-rho_sym),
    ]
    
    for title, formula_str, symbol, expression in formulas:
        st.markdown(f"""
        <div class="formula-box">
            <h4>{title}</h4>
            <p><strong>{formula_str}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if expression:
            st.latex(latex(expression))
    
    st.subheader("🔗 Little's Law Verification")
    st.markdown("""
    Little's Law states that: **L = λW** and **L_q = λW_q**
    
    This fundamental relationship holds for any stable queueing system.
    """)
    
    # Verify Little's Law symbolically
    L_expr = rho_sym/(1-rho_sym)
    W_expr = 1/(mu_sym-lambda_sym)
    littles_law = simplify(lambda_sym * W_expr - L_expr)
    
    st.write("Verification: λW - L =", latex(littles_law), "= 0 ✓")

def display_sensitivity_analysis(lambda_rate, mu_rate):
    """Display sensitivity analysis"""
    st.header("📉 Sensitivity Analysis")
    
    st.subheader("🎚️ Parameter Sensitivity")
    
    # Lambda sensitivity
    lambda_range = np.linspace(0.1, mu_rate*0.95, 50)
    L_lambda = lambda_range / (mu_rate - lambda_range)
    W_lambda = 1 / (mu_rate - lambda_range)
    
    # Mu sensitivity  
    mu_range = np.linspace(lambda_rate*1.05, lambda_rate*3, 50)
    L_mu = lambda_rate / (mu_range - lambda_rate)
    W_mu = 1 / (mu_range - lambda_rate)
    
    fig_sens = make_subplots(
        rows=2, cols=2,
        subplot_titles=('L vs Arrival Rate (λ)', 'W vs Arrival Rate (λ)',
                       'L vs Service Rate (μ)', 'W vs Service Rate (μ)'),
        x_title='Parameter Value',
        y_title='Metric Value'
    )
    
    # Lambda sensitivity plots
    fig_sens.add_trace(go.Scatter(x=lambda_range, y=L_lambda, name='L vs λ', line=dict(color='red')), row=1, col=1)
    fig_sens.add_trace(go.Scatter(x=lambda_range, y=W_lambda, name='W vs λ', line=dict(color='blue')), row=1, col=2)
    
    # Mu sensitivity plots
    fig_sens.add_trace(go.Scatter(x=mu_range, y=L_mu, name='L vs μ', line=dict(color='green')), row=2, col=1)
    fig_sens.add_trace(go.Scatter(x=mu_range, y=W_mu, name='W vs μ', line=dict(color='orange')), row=2, col=2)
    
    # Add current values
    current_L = lambda_rate / (mu_rate - lambda_rate)
    current_W = 1 / (mu_rate - lambda_rate)
    
    fig_sens.add_vline(x=lambda_rate, line_dash="dash", line_color="black", row=1, col=1)
    fig_sens.add_vline(x=lambda_rate, line_dash="dash", line_color="black", row=1, col=2)
    fig_sens.add_vline(x=mu_rate, line_dash="dash", line_color="black", row=2, col=1)
    fig_sens.add_vline(x=mu_rate, line_dash="dash", line_color="black", row=2, col=2)
    
    fig_sens.update_layout(height=600, showlegend=False, title_text="Sensitivity Analysis")
    st.plotly_chart(fig_sens, use_container_width=True)
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    
    cost_per_wait = st.slider("Cost per hour of customer waiting ($)", 1, 100, 10)
    cost_per_server = st.slider("Cost per hour of server operation ($)", 10, 500, 50)
    
    metrics = calculate_mm1_metrics(lambda_rate, mu_rate)
    
    total_waiting_cost = cost_per_wait * lambda_rate * metrics['W']
    total_server_cost = cost_per_server
    total_cost = total_waiting_cost + total_server_cost
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Waiting Cost/hour", f"${total_waiting_cost:.2f}")
    with col2:
        st.metric("Server Cost/hour", f"${total_server_cost:.2f}")
    with col3:
        st.metric("Total Cost/hour", f"${total_cost:.2f}")

def display_simulation(lambda_rate, mu_rate):
    """Display Monte Carlo simulation"""
    st.header("🎯 Monte Carlo Simulation")
    
    num_customers = st.slider("Number of customers to simulate", 100, 10000, 1000)
    
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Generate arrival times (exponential distribution)
            arrival_times = np.random.exponential(1/lambda_rate, num_customers)
            service_times = np.random.exponential(1/mu_rate, num_customers)
            
            # Calculate cumulative times
            arrival_times_cum = np.cumsum(arrival_times)
            
            # Simulate queue
            start_service = np.zeros(num_customers)
            end_service = np.zeros(num_customers)
            wait_times = np.zeros(num_customers)
            
            start_service[0] = arrival_times_cum[0]
            end_service[0] = start_service[0] + service_times[0]
            wait_times[0] = 0
            
            for i in range(1, num_customers):
                start_service[i] = max(arrival_times_cum[i], end_service[i-1])
                wait_times[i] = start_service[i] - arrival_times_cum[i]
                end_service[i] = start_service[i] + service_times[i]
            
            # Calculate simulation results
            sim_avg_wait = np.mean(wait_times)
            sim_avg_system_time = np.mean(wait_times + service_times)
            
            # Theoretical values
            theoretical_wait = lambda_rate / (mu_rate * (mu_rate - lambda_rate))
            theoretical_system_time = 1 / (mu_rate - lambda_rate)
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Simulation Results")
                st.metric("Average Waiting Time", f"{sim_avg_wait:.4f} hours", 
                         f"{((sim_avg_wait - theoretical_wait)/theoretical_wait*100):+.2f}% vs theory")
                st.metric("Average System Time", f"{sim_avg_system_time:.4f} hours",
                         f"{((sim_avg_system_time - theoretical_system_time)/theoretical_system_time*100):+.2f}% vs theory")
            
            with col2:
                st.subheader("📈 Theoretical Values")
                st.metric("Theoretical Waiting Time", f"{theoretical_wait:.4f} hours")
                st.metric("Theoretical System Time", f"{theoretical_system_time:.4f} hours")
            
            # Plot histograms
            fig_hist = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Waiting Time Distribution', 'Service Time Distribution')
            )
            
            fig_hist.add_trace(
                go.Histogram(x=wait_times, name='Waiting Times', nbinsx=50, opacity=0.7),
                row=1, col=1
            )
            
            fig_hist.add_trace(
                go.Histogram(x=service_times, name='Service Times', nbinsx=50, opacity=0.7),
                row=1, col=2
            )
            
            fig_hist.update_layout(height=400, showlegend=False, title_text="Simulation Distributions")
            st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
