"""
Visualization Module - Institutional Grade
Create publication-quality charts for VP capital cycle analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set institutional-grade style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']


def create_all_visualizations(df: pd.DataFrame, macro_df: pd.DataFrame, output_dir: Path) -> None:
    """Create all institutional-grade visualizations"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING INSTITUTIONAL VISUALIZATIONS")
    logger.info("="*80 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== REGIME ANALYSIS ==========
    if not macro_df.empty:
        create_enhanced_regime_heatmap(macro_df, output_dir / "regime_continuous.png")
        create_regime_probability_chart(macro_df, output_dir / "regime_probability.png")
    
    # ========== CAPITAL CYCLE ==========
    create_phase_space_diagram(df, output_dir / "phase_space_trajectory.png")
    create_capital_cycle_diagram(df, output_dir / "capital_cycle_traditional.png")
    
    # ========== CYCLE DECOMPOSITION ==========
    create_cycle_decomposition_chart(df, output_dir / "cycle_decomposition.png")
    
    # ========== INVESTMENT DYNAMICS ==========
    create_intensity_evolution(df, output_dir / "intensity_evolution.png")
    
    # ========== RETURNS & EP ==========
    create_ep_persistence(df, output_dir / "ep_persistence.png")
    create_roic_evolution(df, output_dir / "roic_evolution.png")
    
    # ========== CREDIT MONETIZATION ==========
    create_credit_scurve_chart(df, output_dir / "credit_scurve.png")
    create_credit_charts(df, output_dir)
    
    logger.info("="*80)
    logger.info("✓ ALL VISUALIZATIONS COMPLETE")
    logger.info("="*80 + "\n")


def create_enhanced_regime_heatmap(macro_df: pd.DataFrame, output_path: Path) -> None:
    """
    Continuous scarcity index heatmap with gradient visualization
    """
    logger.info("Creating enhanced regime heatmap...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Top panel: Continuous scarcity index
    if 'scarcity_index_normalized' in macro_df.columns:
        pivot_continuous = macro_df.pivot_table(
            index='country',
            columns=macro_df['date'].dt.to_period('Q'),
            values='scarcity_index_normalized',
            aggfunc='mean'
        )
        
        if not pivot_continuous.empty:
            sns.heatmap(
                pivot_continuous,
                cmap='RdYlGn_r',
                center=50,
                vmin=0,
                vmax=100,
                cbar_kws={'label': 'Capital Scarcity Index (0=Abundant, 100=Severe)'},
                ax=axes[0],
                linewidths=0.3,
                annot=False
            )
            axes[0].set_title('Continuous Capital Scarcity Index (Gradient)', 
                            fontsize=14, fontweight='bold', pad=15)
            axes[0].set_xlabel('')
            axes[0].set_ylabel('Country', fontsize=11, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, 'No continuous scarcity data', 
                        ha='center', va='center', transform=axes[0].transAxes)
    
    # Bottom panel: Regime probability
    if 'prob_scarcity' in macro_df.columns:
        pivot_prob = macro_df.pivot_table(
            index='country',
            columns=macro_df['date'].dt.to_period('Q'),
            values='prob_scarcity',
            aggfunc='mean'
        )
        
        if not pivot_prob.empty:
            sns.heatmap(
                pivot_prob,
                cmap='Reds',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'P(Capital Scarcity Regime)'},
                ax=axes[1],
                linewidths=0.3,
                annot=False
            )
            axes[1].set_title('Scarcity Regime Probability (Continuous)', 
                            fontsize=14, fontweight='bold', pad=15)
            axes[1].set_xlabel('Quarter', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Country', fontsize=11, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No probability data', 
                        ha='center', va='center', transform=axes[1].transAxes)
    
    # Rotate x-axis labels
    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Capital Scarcity Regime Analysis (Institutional Grade)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_regime_probability_chart(macro_df: pd.DataFrame, output_path: Path) -> None:
    """
    Time series of regime probability
    """
    logger.info("Creating regime probability chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'prob_scarcity' in macro_df.columns:
        for country in macro_df['country'].unique():
            country_data = macro_df[macro_df['country'] == country]
            ax.plot(country_data['date'], country_data['prob_scarcity'], 
                   marker='o', linewidth=2, markersize=5, label=country, alpha=0.8)
        
        # Reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='50% Threshold')
        ax.axhline(y=0.75, color='darkred', linestyle=':', linewidth=1.5, 
                  alpha=0.5, label='75% Severe')
        
        ax.fill_between(macro_df['date'].unique(), 0.5, 1.0, alpha=0.1, color='red')
        ax.fill_between(macro_df['date'].unique(), 0, 0.5, alpha=0.1, color='green')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(Capital Scarcity Regime)', fontsize=12, fontweight='bold')
    ax.set_title('Capital Scarcity Regime Probability Over Time', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_phase_space_diagram(df: pd.DataFrame, output_path: Path) -> None:
    """
    Capital cycle phase diagram with trajectory and regime overlay
    This is the flagship institutional chart
    """
    logger.info("Creating phase space trajectory diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get valid data
    plot_df = df[
        df['phase_x'].notna() & df['phase_y'].notna()
    ].sort_values('date').copy()
    
    if len(plot_df) < 2:
        ax.text(0.5, 0.5, 'Insufficient data for phase space diagram', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Color by scarcity index or time
    if 'scarcity_index_normalized' in plot_df.columns and plot_df['scarcity_index_normalized'].notna().any():
        colors = plot_df['scarcity_index_normalized']
        cmap = 'RdYlGn_r'
        label = 'Capital Scarcity Index'
        vmin, vmax = 0, 100
    else:
        colors = range(len(plot_df))
        cmap = 'viridis'
        label = 'Time Progression'
        vmin, vmax = None, None
    
    # Main scatter with trajectory
    scatter = ax.scatter(
        plot_df['phase_x'],
        plot_df['phase_y'],
        c=colors,
        s=200,
        cmap=cmap,
        alpha=0.8,
        edgecolors='black',
        linewidth=2,
        zorder=3,
        vmin=vmin,
        vmax=vmax
    )
    
    # Connect trajectory
    ax.plot(
        plot_df['phase_x'],
        plot_df['phase_y'],
        color='black',
        alpha=0.4,
        linewidth=2.5,
        zorder=1,
        linestyle='-'
    )
    
    # Add directional arrows (every few points)
    arrow_interval = max(1, len(plot_df) // 8)
    for i in range(0, len(plot_df)-1, arrow_interval):
        if i + 1 < len(plot_df):
            ax.annotate(
                '',
                xy=(plot_df.iloc[i+1]['phase_x'], plot_df.iloc[i+1]['phase_y']),
                xytext=(plot_df.iloc[i]['phase_x'], plot_df.iloc[i]['phase_y']),
                arrowprops=dict(
                    arrowstyle='->', 
                    lw=2, 
                    color='black', 
                    alpha=0.6,
                    mutation_scale=20
                ),
                zorder=2
            )
    
    # Quadrant reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6, zorder=0)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.6, zorder=0)
    
    # Quadrant labels with VP interpretation
    ax.text(0.02, 0.98, 'VALUE CREATION\n(Low Investment)', 
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgreen', 
                     alpha=0.7, edgecolor='darkgreen', linewidth=2))
    
    ax.text(0.98, 0.98, 'SUPER-CYCLE\n(High ROIC + High Investment)', 
            transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='gold', 
                     alpha=0.8, edgecolor='orange', linewidth=2))
    
    ax.text(0.02, 0.02, 'CAPITAL ABUNDANCE\n(Low Returns)', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', 
                     alpha=0.6, edgecolor='blue', linewidth=2))
    
    ax.text(0.98, 0.02, 'CAPITAL DESTRUCTION\n(High Investment, Low ROIC)', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='red', 
                     alpha=0.7, edgecolor='darkred', linewidth=2))
    
    # Mark key points
    if len(plot_df) >= 2:
        first = plot_df.iloc[0]
        last = plot_df.iloc[-1]
        
        # Start point
        ax.scatter(first['phase_x'], first['phase_y'], 
                  s=400, marker='s', c='blue', edgecolors='black', 
                  linewidth=3, zorder=5, label='Start')
        ax.annotate(f"START\n{first['date'].strftime('%Y-Q%q')}", 
                   xy=(first['phase_x'], first['phase_y']),
                   xytext=(15, 15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
        
        # End point
        ax.scatter(last['phase_x'], last['phase_y'], 
                  s=500, marker='*', c='gold', edgecolors='black', 
                  linewidth=3, zorder=5, label='Latest')
        ax.annotate(f"LATEST\n{last['date'].strftime('%Y-Q%q')}", 
                   xy=(last['phase_x'], last['phase_y']),
                   xytext=(15, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', fc='yellow', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, rotation=270, labelpad=25, fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Investment Intensity (Z-Score)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ROIC (Z-Score)', fontsize=13, fontweight='bold')
    ax.set_title('Capital Cycle Phase Space Trajectory\n(VP Institutional Framework)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_cycle_decomposition_chart(df: pd.DataFrame, output_path: Path) -> None:
    """
    Show structural trend vs cyclical component for ROIC and Investment
    """
    logger.info("Creating cycle decomposition chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # ROIC decomposition
    if 'roic_trend' in df.columns and df['roic_trend'].notna().any():
        # Top-left: ROIC trend
        ax = axes[0, 0]
        ax.plot(df['date'], df['roic'], 
               label='Actual ROIC', color='gray', alpha=0.5, linewidth=1.5)
        ax.plot(df['date'], df['roic_trend'], 
               label='Structural Trend (HP Filter)', color='#2E86AB', linewidth=3)
        ax.set_ylabel('ROIC (%)', fontsize=11, fontweight='bold')
        ax.set_title('ROIC: Structural Trend', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Top-right: ROIC cycle
        ax = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df['roic_cycle'].fillna(0)]
        ax.bar(df['date'], df['roic_cycle'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=2)
        ax.set_ylabel('Cyclical Component (%)', fontsize=11, fontweight='bold')
        ax.set_title('ROIC: Cyclical Deviation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        for ax in axes[0, :]:
            ax.text(0.5, 0.5, 'ROIC decomposition unavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Investment decomposition
    if 'total_capex_intensity_trend' in df.columns and df['total_capex_intensity_trend'].notna().any():
        # Bottom-left: Investment trend
        ax = axes[1, 0]
        ax.plot(df['date'], df['total_capex_intensity'], 
               label='Actual Investment', color='gray', alpha=0.5, linewidth=1.5)
        ax.plot(df['date'], df['total_capex_intensity_trend'], 
               label='Structural Trend (HP Filter)', color='#06A77D', linewidth=3)
        ax.set_ylabel('Investment Intensity (%)', fontsize=11, fontweight='bold')
        ax.set_title('Investment Intensity: Structural Trend', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Investment cycle
        ax = axes[1, 1]
        colors = ['green' if x > 0 else 'red' for x in df['total_capex_intensity_cycle'].fillna(0)]
        ax.bar(df['date'], df['total_capex_intensity_cycle'], 
              color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=2)
        ax.set_ylabel('Cyclical Component (%)', fontsize=11, fontweight='bold')
        ax.set_title('Investment Intensity: Cyclical Deviation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        for ax in axes[1, :]:
            ax.text(0.5, 0.5, 'Investment decomposition unavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # Format all x-axes
    for ax in axes.flat:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('HP Filter Decomposition: Trend vs Cycle', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_credit_scurve_chart(df: pd.DataFrame, output_path: Path) -> None:
    """
    Credit monetization S-curve with log scaling and inflection detection
    """
    logger.info("Creating credit S-curve chart...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Panel 1: Log scale credit portfolio
    ax1 = axes[0]
    if 'credit_portfolio_log' in df.columns and df['credit_portfolio_log'].notna().any():
        ax1.plot(df['date'], df['credit_portfolio_log'], 
                marker='o', linewidth=2.5, markersize=6, color='#2E86AB', 
                label='Log(Credit Portfolio)')
        
        # Polynomial fit to show S-curve
        valid = df[df['credit_portfolio_log'].notna()]
        if len(valid) > 5:
            z = np.polyfit(range(len(valid)), valid['credit_portfolio_log'], 2)
            p = np.poly1d(z)
            ax1.plot(valid['date'], p(range(len(valid))), 
                    '--', color='red', linewidth=2.5, label='Quadratic Fit', alpha=0.7)
    
    ax1.set_ylabel('Log(Credit Portfolio)', fontsize=12, fontweight='bold')
    ax1.set_title('Credit Monetization S-Curve (Log Scale)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Log growth rate
    ax2 = axes[1]
    if 'credit_portfolio_log_growth' in df.columns:
        ax2.plot(df['date'], df['credit_portfolio_log_growth'], 
                marker='s', linewidth=2, markersize=5, color='#F18F01', 
                label='Log Growth Rate')
        ax2.axhline(0, color='black', linewidth=1.5, alpha=0.5)
        ax2.fill_between(df['date'], 0, df['credit_portfolio_log_growth'], 
                        alpha=0.3, color='#F18F01')
    
    ax2.set_ylabel('Growth Rate (% QoQ)', fontsize=12, fontweight='bold')
    ax2.set_title('Credit Growth Rate', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Growth acceleration (inflection detection)
    ax3 = axes[2]
    if 'credit_portfolio_log_accel' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['credit_portfolio_log_accel'].fillna(0)]
        ax3.bar(df['date'], df['credit_portfolio_log_accel'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axhline(0, color='black', linewidth=2)
        
        # Mark significant inflection points
        inflections = df[np.abs(df['credit_portfolio_log_accel']) > df['credit_portfolio_log_accel'].std()]
        for idx in inflections.index:
            ax3.axvline(df.loc[idx, 'date'], color='purple', linestyle='--', 
                       linewidth=2, alpha=0.6, label='Inflection' if idx == inflections.index[0] else '')
    
    ax3.set_ylabel('Growth Acceleration', fontsize=12, fontweight='bold')
    ax3.set_title('Credit Growth Acceleration (Inflection Detection)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axes
    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Credit Portfolio: S-Curve Analysis (Institutional)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_capital_cycle_diagram(df: pd.DataFrame, output_path: Path) -> None:
    """
    Traditional capital cycle scatter (simplified, no arrows)
    """
    logger.info("Creating traditional capital cycle diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_df = df[
        df['roic'].notna() & 
        df['total_capex_intensity'].notna()
    ].copy()
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, 'Insufficient data\n(Need ROIC and Capex Intensity)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlabel('Total Capex Intensity (%)', fontsize=12)
        ax.set_ylabel('ROIC (%)', fontsize=12)
        ax.set_title('Capital Cycle Position', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plot_df = plot_df.sort_values('date').reset_index(drop=True)
    plot_df['time_order'] = range(len(plot_df))
    
    # Scatter with time gradient
    scatter = ax.scatter(
        plot_df['total_capex_intensity'],
        plot_df['roic'],
        c=plot_df['time_order'],
        s=150,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5,
        zorder=3
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Time Progression →', rotation=270, labelpad=20, fontsize=11)
    
    # Reference lines
    wacc = 17
    if plot_df['total_capex_intensity'].notna().any():
        median_capex = plot_df['total_capex_intensity'].median()
        ax.axvline(x=median_capex, color='gray', linestyle='--', linewidth=1.5, 
                   alpha=0.5, label=f'Median Capex ({median_capex:.1f}%)', zorder=1)
    
    ax.axhline(y=wacc, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.5, label=f'WACC ({wacc}%)', zorder=1)
    
    # Quadrant labels
    ax.text(0.02, 0.98, 'VALUE CREATION\nLow Investment', 
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    ax.text(0.98, 0.98, 'VALUE CREATION\nHigh Investment', 
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    # Mark start and end
    if len(plot_df) >= 2:
        first = plot_df.iloc[0]
        last = plot_df.iloc[-1]
        
        ax.scatter(first['total_capex_intensity'], first['roic'], 
                  s=200, marker='s', c='blue', edgecolors='black', linewidth=2, 
                  zorder=4, label='Start')
        
        ax.scatter(last['total_capex_intensity'], last['roic'], 
                  s=250, marker='*', c='gold', edgecolors='black', linewidth=2, 
                  zorder=4, label='Latest')
    
    ax.set_xlabel('Total Capex Intensity (% of Revenue)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROIC (%)', fontsize=12, fontweight='bold')
    ax.set_title('Capital Cycle Position (Traditional View)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_intensity_evolution(df: pd.DataFrame, output_path: Path) -> None:
    """Three-panel intensity evolution chart"""
    logger.info("Creating intensity evolution chart...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel 1: Growth Opex
    ax1 = axes[0]
    if 'growth_opex_intensity' in df.columns:
        ax1.plot(df['date'], df['growth_opex_intensity'], 
                marker='o', linewidth=2.5, markersize=6, color='#2E86AB', label='Growth Opex Intensity')
        ax1.fill_between(df['date'], 0, df['growth_opex_intensity'], alpha=0.3, color='#2E86AB')
    
    if 'rd_intensity' in df.columns:
        ax1.plot(df['date'], df['rd_intensity'], 
                marker='s', linewidth=1.5, markersize=4, color='#A23B72', 
                linestyle='--', label='R&D', alpha=0.7)
    
    if 'sm_intensity' in df.columns:
        ax1.plot(df['date'], df['sm_intensity'], 
                marker='^', linewidth=1.5, markersize=4, color='#F18F01', 
                linestyle='--', label='S&M', alpha=0.7)
    
    ax1.set_ylabel('% of Revenue', fontsize=11, fontweight='bold')
    ax1.set_title('Growth Opex Investment Intensity', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Capex
    ax2 = axes[1]
    if 'total_capex_intensity' in df.columns:
        ax2.plot(df['date'], df['total_capex_intensity'], 
                marker='o', linewidth=2.5, markersize=6, color='#06A77D', label='Total Capex')
        ax2.fill_between(df['date'], 0, df['total_capex_intensity'], alpha=0.3, color='#06A77D')
    
    ax2.set_ylabel('% of Revenue', fontsize=11, fontweight='bold')
    ax2.set_title('Capital Expenditure Intensity', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Total Growth Investment
    ax3 = axes[2]
    if 'growth_investment_intensity' in df.columns:
        ax3.plot(df['date'], df['growth_investment_intensity'], 
                marker='o', linewidth=2.5, markersize=7, color='#C73E1D', label='Total Growth Investment')
        ax3.fill_between(df['date'], 0, df['growth_investment_intensity'], alpha=0.3, color='#C73E1D')
        ax3.axhline(y=30, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='30% Threshold')
    
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('% of Revenue', fontsize=11, fontweight='bold')
    ax3.set_title('Total Growth Investment (Opex + Capex)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.suptitle('Investment Intensity Evolution', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_roic_evolution(df: pd.DataFrame, output_path: Path) -> None:
    """
    ROIC evolution with trend and incremental ROIC
    """
    logger.info("Creating ROIC evolution chart...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top: ROIC with trend
    if 'roic' in df.columns:
        ax1.plot(df['date'], df['roic'], 
                marker='o', linewidth=2, markersize=5, color='gray', 
                alpha=0.5, label='Actual ROIC')
        
        if 'roic_trend' in df.columns:
            ax1.plot(df['date'], df['roic_trend'], 
                    linewidth=3, color='#2E86AB', label='Structural Trend')
        
        # WACC reference
        ax1.axhline(y=17, color='red', linestyle='--', linewidth=2, 
                   alpha=0.6, label='WACC (17%)')
    
    ax1.set_ylabel('ROIC (%)', fontsize=12, fontweight='bold')
    ax1.set_title('ROIC Evolution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Incremental ROIC
    if 'incremental_roic' in df.columns:
        ax2.plot(df['date'], df['incremental_roic'], 
                marker='s', linewidth=2.5, markersize=6, color='#06A77D', 
                label='Incremental ROIC')
        ax2.axhline(y=17, color='red', linestyle='--', linewidth=2, 
                   alpha=0.6, label='WACC')
        ax2.fill_between(df['date'], 17, df['incremental_roic'], 
                        where=df['incremental_roic'] > 17, 
                        alpha=0.3, color='green', label='Value Creation')
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Incremental ROIC (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Incremental ROIC (3Y Rolling)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.suptitle('Returns Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_ep_persistence(df: pd.DataFrame, output_path: Path) -> None:
    """Economic profit persistence chart"""
    logger.info("Creating EP persistence chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'economic_profit_ttm' in df.columns:
        ax.bar(df['date'], df['economic_profit_ttm'], 
              color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        if 'ep_margin' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df['date'], df['ep_margin'], 
                    color='#F18F01', marker='o', linewidth=2.5, markersize=6, 
                    label='EP Margin (%)')
            ax2.set_ylabel('EP Margin (% of Revenue)', fontsize=11, fontweight='bold', color='#F18F01')
            ax2.tick_params(axis='y', labelcolor='#F18F01')
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(False)
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Economic Profit (TTM, $M)', fontsize=11, fontweight='bold')
    ax.set_title('Economic Profit Persistence', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: {output_path.name}")


def create_credit_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Create credit portfolio charts"""
    logger.info("Creating credit portfolio charts...")
    
    # Chart 1: Credit Growth
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'credit_growth' in df.columns:
        colors = ['green' if x > 0 else 'red' for x in df['credit_growth'].fillna(0)]
        ax.bar(df['date'], df['credit_growth'], color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        
        if 'credit_growth_accel' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(df['date'], df['credit_growth_accel'], 
                    color='#2E86AB', marker='o', linewidth=2, markersize=5, 
                    label='Growth Acceleration')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_ylabel('Growth Acceleration (pp)', fontsize=11, fontweight='bold', color='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.legend(loc='upper left', fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Credit Growth YoY (%)', fontsize=11, fontweight='bold')
    ax.set_title('Credit Portfolio Growth Dynamics', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "credit_growth.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Credit Intensity
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'credit_intensity' in df.columns:
        ax.plot(df['date'], df['credit_intensity'], 
               marker='o', linewidth=2.5, markersize=7, color='#A23B72', label='Credit Intensity')
        ax.fill_between(df['date'], 0, df['credit_intensity'], alpha=0.3, color='#A23B72')
        ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label='20% Strategic Threshold')
    
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Credit Portfolio (% of Revenue)', fontsize=11, fontweight='bold')
    ax.set_title('Credit Deployment Intensity', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "credit_intensity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved: credit_growth.png, credit_intensity.png")


def create_summary_table(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Create summary metrics table"""
    logger.info("Creating summary table...")
    
    latest = df.iloc[-1] if len(df) > 0 else None
    
    if latest is None:
        logger.warning("No data for summary table")
        return pd.DataFrame()
    
    def fmt(val, decimals=1):
        if pd.isna(val):
            return "N/A"
        return f"{val:.{decimals}f}"
    
    summary_data = {
        'Metric': [
            '━━━ RETURNS ━━━',
            'ROIC (TTM, %)',
            'Incremental ROIC (3Y, %)',
            'ROIC vs WACC (bps)',
            'Economic Profit (TTM, $M)',
            'EP Margin (%)',
            '',
            '━━━ INVESTED CAPITAL ━━━',
            'Invested Capital ($M)',
            '- Net Working Capital ($M)',
            '- Gross PP&E ($M)',
            '- Capitalized R&D ($M)',
            '',
            '━━━ CAPEX INTENSITY ━━━',
            'Growth Capex Intensity (%)',
            'Maintenance Capex Intensity (%)',
            'Total Capex Intensity (%)',
            'Growth Capex Share (%)',
            '',
            '━━━ OPEX INTENSITY ━━━',
            'R&D Intensity (%)',
            'S&M Intensity (%)',
            'Growth Opex Intensity (%)',
            '',
            '━━━ TOTAL GROWTH INVESTMENT ━━━',
            'Total Growth Investment ($M, TTM)',
            'Growth Investment Intensity (%)',
            'Reinvestment Rate (%)',
            '',
            '━━━ CREDIT PORTFOLIO ━━━',
            'Credit Portfolio ($M)',
            'Credit Growth YoY (%)',
            'Credit Growth Acceleration (pp)',
            'Credit Intensity (% Revenue)',
            '',
            '━━━ MACRO REGIME ━━━',
            'Capital Scarcity Index (0-100)',
            'P(Scarcity Regime)',
            'Regime Classification',
            '',
            '━━━ PHASE SPACE ━━━',
            'ROIC Z-Score',
            'Investment Z-Score',
            'Phase Quadrant'
        ],
        'Latest': [
            '',
            fmt(latest.get('roic'), 1),
            fmt(latest.get('incremental_roic'), 1),
            fmt((latest.get('roic', 0) - 17) * 100, 0) if pd.notna(latest.get('roic')) else 'N/A',
            fmt(latest.get('economic_profit_ttm'), 0),
            fmt(latest.get('ep_margin'), 1),
            '',
            '',
            fmt(latest.get('invested_capital'), 0),
            fmt(latest.get('nwc'), 0),
            fmt(latest.get('gross_ppe'), 0),
            fmt(latest.get('capitalized_rd'), 0),
            '',
            '',
            fmt(latest.get('growth_capex_intensity'), 1),
            fmt(latest.get('maintenance_capex_intensity'), 1),
            fmt(latest.get('total_capex_intensity'), 1),
            fmt(latest.get('growth_capex_share'), 1),
            '',
            '',
            fmt(latest.get('rd_intensity'), 1),
            fmt(latest.get('sm_intensity'), 1),
            fmt(latest.get('growth_opex_intensity'), 1),
            '',
            '',
            fmt(latest.get('total_growth_investment_ttm'), 0),
            fmt(latest.get('growth_investment_intensity'), 1),
            fmt(latest.get('reinvestment_rate'), 1),
            '',
            '',
            fmt(latest.get('credit_portfolio'), 0),
            fmt(latest.get('credit_growth'), 1),
            fmt(latest.get('credit_growth_accel'), 1),
            fmt(latest.get('credit_intensity'), 1),
            '',
            '',
            fmt(latest.get('scarcity_index_normalized'), 0),
            fmt(latest.get('prob_scarcity'), 2),
            str(latest.get('regime_probabilistic', 'N/A')),
            '',
            '',
            fmt(latest.get('roic_z'), 2),
            fmt(latest.get('phase_x'), 2),
            str(latest.get('phase_quadrant', 'N/A'))
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"  ✓ Saved: {output_path.name}")
    
    return summary_df