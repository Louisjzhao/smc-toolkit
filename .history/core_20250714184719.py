# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def calc_swing_structures(df: pd.DataFrame, size: int, time_series: pd.Series, prefix: str = '') -> pd.DataFrame:
    # === Identify initial swing high/low points ===
    swing_pre = pd.Series(0, index=df.index)
    swing_pre[df['high'] > df['high'].shift(-size).rolling(size).max()] = 1
    swing_pre[df['low'] < df['low'].shift(-size).rolling(size).min()] = -1
    df[f'{prefix}swing_pre'] = swing_pre

    # === Vectorized construction of swing state sequence ===
    swing_hl_sim = (
        swing_pre.replace(0, np.nan)
                 .ffill()
                 .replace({1: 0, -1: 1})
                 .fillna(0)
    )
    df[f'{prefix}swing_hl_sim'] = swing_hl_sim.astype(int)
    df[f'{prefix}swing_h_l'] = swing_hl_sim.diff()

    # === Price and time of swing high/low ===
    swing_high_level = pd.Series(
        np.where(df[f'{prefix}swing_h_l'] == -1, df['high'], np.nan),
        index=df.index
    ).ffill()

    swing_low_level = pd.Series(
        np.where(df[f'{prefix}swing_h_l'] == 1, df['low'], np.nan),
        index=df.index
    ).ffill()

    swing_high_time = time_series.where(df[f'{prefix}swing_h_l'] == -1).ffill()
    swing_low_time = time_series.where(df[f'{prefix}swing_h_l'] == 1).ffill()

    # === BOS detection ===
    bullish_bos = (df['close'] > swing_high_level) & (df['close'].shift(1) <= swing_high_level)
    bearish_bos = (df['close'] < swing_low_level) & (df['close'].shift(1) >= swing_low_level)

    df[f'{prefix}bos'] = 0
    df.loc[bullish_bos, f'{prefix}bos'] = 1
    df.loc[bearish_bos, f'{prefix}bos'] = -1
    df[f'{prefix}bos'] = df[f'{prefix}bos'].astype('Int64')

    df[f'{prefix}bos_level'] = np.nan
    df.loc[bullish_bos, f'{prefix}bos_level'] = swing_high_level
    df.loc[bearish_bos, f'{prefix}bos_level'] = swing_low_level

    df[f'{prefix}bos_anchor_time'] = pd.NaT
    df.loc[bullish_bos, f'{prefix}bos_anchor_time'] = swing_high_time
    df.loc[bearish_bos, f'{prefix}bos_anchor_time'] = swing_low_time

    df[f'{prefix}bos_tag'] = df[f'{prefix}bos'].map({1: 'bullish', -1: 'bearish'})
    df[f'{prefix}bos_time'] = time_series.where(df[f'{prefix}bos'] != 0)

    # === CHoCH detection ===
    trend = df[f'{prefix}bos'].replace(0, np.nan).ffill()
    trend_shift = trend.diff()

    df[f'{prefix}choch'] = trend_shift.where(trend_shift.isin([-2, 2])).astype('Int64')
    df[f'{prefix}choch_tag'] = df[f'{prefix}choch'].map({2: 'bullish', -2: 'bearish'})

    df[f'{prefix}choch_level'] = np.nan
    df[f'{prefix}choch_anchor_time'] = pd.NaT
    df.loc[df[f'{prefix}choch'] == 2, f'{prefix}choch_level'] = swing_high_level[df[f'{prefix}choch'] == 2]
    df.loc[df[f'{prefix}choch'] == 2, f'{prefix}choch_anchor_time'] = swing_high_time[df[f'{prefix}choch'] == 2]
    df.loc[df[f'{prefix}choch'] == -2, f'{prefix}choch_level'] = swing_low_level[df[f'{prefix}choch'] == -2]
    df.loc[df[f'{prefix}choch'] == -2, f'{prefix}choch_anchor_time'] = swing_low_time[df[f'{prefix}choch'] == -2]

    return df

def process_smc_with_internal(df: pd.DataFrame, swing_size: int = 40, internal_size: int = 5) -> pd.DataFrame:
    # Apply both main and internal SMC structure detection
    df = df.sort_index(level='date')
    time_series = pd.Series(df.index.get_level_values('date'), index=df.index)

    df = calc_swing_structures(df, swing_size, time_series, prefix='')
    df = calc_swing_structures(df, internal_size, time_series, prefix='int_')

    return df

def extract_ob_blocks(df: pd.DataFrame) -> pd.DataFrame:
    # Extract order blocks based on CHoCH patterns
    ob_data = []

    for time, row in df[df['choch'].notna()].iterrows():
        _, end_time = time
        start_time = row['choch_anchor_time']

        if pd.isna(start_time) or start_time >= end_time:
            continue

        segment = df.loc[(df.index.get_level_values(1) >= start_time) &
                         (df.index.get_level_values(1) <= end_time)]

        if segment.empty:
            continue

        if row['choch'] == 2:
            ob_index = segment['low'].idxmin()
            ob_row = df.loc[ob_index]
            ob_top = ob_row['high']
            ob_bottom = ob_row['low']
            ob_type = 'bullish'
        elif row['choch'] == -2:
            ob_index = segment['high'].idxmax()
            ob_row = df.loc[ob_index]
            ob_top = ob_row['high']
            ob_bottom = ob_row['low']
            ob_type = 'bearish'
        else:
            continue

        ob_data.append({
            'code': ob_index[0],
            'ob_time': ob_index[1],
            'start_time': start_time,
            'end_time': end_time,
            'choch_time': end_time,
            'ob_top': ob_top,
            'ob_bottom': ob_bottom,
            'ob_type': ob_type
        })

    return pd.DataFrame(ob_data)

def extract_fvg(df: pd.DataFrame) -> pd.DataFrame:
    # Extract fair value gaps (FVGs) from price gaps and price expansion conditions
    ts = pd.Series(df.index.get_level_values(1), index=df.index)

    bar_delta_percent = (df['close'].shift(1) - df['open'].shift(1)) / df['open'].shift(1) / 100
    threshold = bar_delta_percent.abs().expanding().mean() * 2

    bullish_mask = (
        (df['low'] > df['high'].shift(2)) &
        (df['close'].shift(1) > df['high'].shift(2)) &
        (bar_delta_percent > threshold)
    )

    bearish_mask = (
        (df['high'] < df['low'].shift(2)) &
        (df['close'].shift(1) < df['low'].shift(2)) &
        (-bar_delta_percent > threshold)
    )

    fvg_df = pd.DataFrame(index=df.index)
    fvg_df['fvg_type'] = np.select(
        [bullish_mask, bearish_mask],
        ['bullish', 'bearish'],
        default=None
    )

    fvg_df['fvg_top'] = np.where(
        fvg_df['fvg_type'] == 'bullish', df['low'],
        np.where(fvg_df['fvg_type'] == 'bearish', df['low'].shift(2), np.nan)
    )

    fvg_df['fvg_bottom'] = np.where(
        fvg_df['fvg_type'] == 'bullish', df['high'].shift(2),
        np.where(fvg_df['fvg_type'] == 'bearish', df['high'], np.nan)
    )

    fvg_df['start_time'] = ts.shift(2)
    fvg_df['end_time'] = ts
    fvg_df['fvg_size'] = fvg_df['fvg_top'] - fvg_df['fvg_bottom']
    fvg_df['code'] = df.index.get_level_values(0)[0]

    fvg_df['fvg_mid'] = (fvg_df['fvg_top'] + fvg_df['fvg_bottom']) / 2
    fvg_df['mitigated'] = False

    fvg_df = fvg_df.dropna(subset=['fvg_type']).copy()

    for i, row in fvg_df.iterrows():
        end_time = row['end_time']
        mid = row['fvg_mid']
        fvg_type = row['fvg_type']

        future_df = df[df.index.get_level_values(1) > end_time]

        if fvg_type == 'bullish':
            condition = future_df['low'] <= mid
        else:
            condition = future_df['high'] >= mid

        if condition.any():
            fvg_df.at[i, 'mitigated'] = True

    return fvg_df.reset_index(drop=True)



def tag_fvg(df: pd.DataFrame, fvg_df: pd.DataFrame) -> pd.DataFrame:
    """Tag bars inside bullish or bearish Fair Value Gaps (FVGs)."""
    df['in_fvg_bull'] = False
    df['in_fvg_bear'] = False

    for _, row in fvg_df.iterrows():
        time_mask = (
            (df.index.get_level_values(1) >= row['start_time']) &
            (df.index.get_level_values(1) <= row['end_time'])
        )
        price_mask = (
            (df['low'] < row['fvg_top']) &
            (df['high'] > row['fvg_bottom'])
        )

        if row['fvg_type'] == 'bullish':
            df.loc[time_mask & price_mask, 'in_fvg_bull'] = True
        elif row['fvg_type'] == 'bearish':
            df.loc[time_mask & price_mask, 'in_fvg_bear'] = True

    return df


def plot_smc_structure(code, result_df, ob_df=None, fvg_df=None, show_internal=True):
    """Plot SMC structure including BOS, CHoCH, OB, FVG, and swing points."""
    df_vis = result_df.loc[code].copy()
    df_vis.reset_index(inplace=True)
    df_vis.index = df_vis['date']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"SMC Structure: BOS & CHoCH [{code}]")

    # === Candlestick Chart ===
    candles = df_vis[['open', 'high', 'low', 'close']]
    for time, row in candles.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.plot([time, time], [row['low'], row['high']], color='black', linewidth=1)
        ax1.add_patch(plt.Rectangle((time - pd.Timedelta(hours=6), min(row['open'], row['close'])),
                                    pd.Timedelta(hours=12),
                                    abs(row['close'] - row['open']),
                                    color=color))

    # === BOS lines ===
    bos_rows = df_vis[df_vis['bos'] != 0]
    for time, row in bos_rows.iterrows():
        anchor_time = row['bos_anchor_time']
        if pd.notna(anchor_time):
            ax1.hlines(row['bos_level'], xmin=anchor_time, xmax=time,
                       colors='blue' if row['bos'] == 1 else 'orange',
                       linestyles='dashed', linewidth=1.2)
            ax1.text(time, row['bos_level'], 'BOS',
                     color='blue' if row['bos'] == 1 else 'orange',
                     fontsize=8, verticalalignment='bottom')

    # === CHoCH lines ===
    choch_rows = df_vis[df_vis['choch'] != 0]
    for time, row in choch_rows.iterrows():
        anchor_time = row['choch_anchor_time']
        if pd.notna(anchor_time):
            ax1.hlines(row['choch_level'], xmin=anchor_time, xmax=time,
                       colors='purple' if row['choch'] == 2 else 'brown',
                       linestyles='solid', linewidth=1.5)
            ax1.text(time, row['choch_level'], 'CHoCH',
                     color='purple' if row['choch'] == 2 else 'brown',
                     fontsize=8, verticalalignment='top')

    # === Swing High / Low markers ===
    swh_label = swl_label = False
    for time, row in df_vis.iterrows():
        if row['swing_h_l'] == -1:
            ax1.plot(time, row['high'], marker='v', color='blue', markersize=8,
                     label='Swing High' if not swh_label else "")
            swh_label = True
        elif row['swing_h_l'] == 1:
            ax1.plot(time, row['low'], marker='^', color='orange', markersize=8,
                     label='Swing Low' if not swl_label else "")
            swl_label = True

    # === Internal Swing High / Low ===
    if show_internal:
        ish_label = isl_label = False
        for time, row in df_vis.iterrows():
            if row.get('int_swing_h_l') == -1:
                ax1.plot(time, row['high'], marker='v', color='gray', markersize=6,
                         label='Internal High' if not ish_label else "")
                ish_label = True
            elif row.get('int_swing_h_l') == 1:
                ax1.plot(time, row['low'], marker='^', color='gray', markersize=6,
                         label='Internal Low' if not isl_label else "")
                isl_label = True

        # Internal BOS
        int_bos_rows = df_vis[df_vis['int_bos'].fillna(0) != 0]
        for time, row in int_bos_rows.iterrows():
            anchor_time = row['int_bos_anchor_time']
            if pd.notna(anchor_time):
                ax1.hlines(row['int_bos_level'], xmin=anchor_time, xmax=time,
                           colors='gray', linestyles='dashed', linewidth=1)
                ax1.text(time, row['int_bos_level'], 'int_BOS',
                         color='gray', fontsize=7, verticalalignment='bottom')

        # Internal CHoCH
        int_choch_rows = df_vis[df_vis['int_choch'].fillna(0) != 0]
        for time, row in int_choch_rows.iterrows():
            anchor_time = row['int_choch_anchor_time']
            if pd.notna(anchor_time):
                ax1.hlines(row['int_choch_level'], xmin=anchor_time, xmax=time,
                           colors='dimgray', linestyles='solid', linewidth=1)
                ax1.text(time, row['int_choch_level'], 'int_CHoCH',
                         color='dimgray', fontsize=7, verticalalignment='top')

    # === OB regions ===
    if ob_df is not None:
        ob_code_df = ob_df[ob_df['code'] == code]
        for _, ob in ob_code_df.iterrows():
            color = 'green' if ob['ob_type'] == 'bullish' else 'red'
            alpha = 0.2
            ax1.axvspan(ob['start_time'], ob['end_time'],
                        ymin=(ob['ob_bottom'] - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                        ymax=(ob['ob_top'] - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                        facecolor=color, alpha=alpha)
            ax1.text(ob['end_time'], ob['ob_top'],
                     f"OB ({ob['ob_type']})", fontsize=8, color=color,
                     verticalalignment='bottom')

    # === FVG regions ===
    if fvg_df is not None:
        fvg_code_df = fvg_df[fvg_df['code'] == code]
        for _, fvg in fvg_code_df.iterrows():
            color = 'blue' if fvg['fvg_type'] == 'bullish' else 'orange'
            alpha = 0.15
            hatch = '//' if fvg.get('mitigated', False) else None
            edgecolor = 'black' if fvg.get('mitigated', False) else None
            label_text = f"FVG ({fvg['fvg_type']})"
            if fvg.get('mitigated', False):
                label_text += " ✓"

            ax1.axvspan(fvg['start_time'], fvg['end_time'],
                        ymin=(fvg['fvg_bottom'] - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                        ymax=(fvg['fvg_top'] - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                        facecolor=color, alpha=alpha,
                        edgecolor=edgecolor, hatch=hatch, linewidth=0.8)

            ax1.text(fvg['end_time'], fvg['fvg_top'],
                     label_text,
                     fontsize=8,
                     color='dimgray' if fvg.get('mitigated', False) else color,
                     verticalalignment='bottom')

    # === Volume bar ===
    if 'volume' in df_vis.columns:
        ax2.bar(df_vis.index, df_vis['volume'], color='grey', width=0.5)
        ax2.set_ylabel("Volume")

    # === Time formatting ===
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    ax1.set_ylabel("Price")
    ax1.grid(True)
    ax2.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

