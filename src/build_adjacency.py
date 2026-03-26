"""
build_adjacency.py
Parse UN Comtrade HS-2709 CSVs into monthly 5x5 trade adjacency matrices.

Column mapping note (Comtrade CSV has shifted headers):
  reporterCode  = ISO3 of reporting country
  partnerCode   = ISO3 of partner country
  reporterDesc  = flow code (X=export, M=import, RX/DX/RM/FM = re-export/re-import)
  refMonth      = period YYYYMM
  fobvalue      = FOB trade value USD
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from config import COMTRADE_DIR, N_NODES

NODE_MAP   = {'USA': 0, 'GBR': 1, 'NOR': 1, 'SAU': 2, 'RUS': 3, 'IND': 4}
NODE_NAMES = {0: 'WTI/USA', 1: 'Brent/UK+NOR', 2: 'OPEC/SAU', 3: 'Urals/RUS', 4: 'Indian/IND'}
VALID_ISO  = set(NODE_MAP.keys())
OUT_PARQ   = COMTRADE_DIR / "adjacency_monthly.parquet"
OUT_CSV    = COMTRADE_DIR / "adjacency_monthly_readable.csv"


def _load_raw() -> pd.DataFrame:
    files = sorted(COMTRADE_DIR.glob("TradeData_*.csv"))
    frames = [pd.read_csv(f, encoding='latin-1',
                          usecols=['reporterCode', 'partnerCode', 'reporterDesc',
                                   'refMonth', 'fobvalue'],
                          low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        'reporterCode': 'reporter',
        'partnerCode':  'partner',
        'reporterDesc': 'flow',
        'refMonth':     'period',
        'fobvalue':     'value',
    })
    df['value']  = pd.to_numeric(df['value'],  errors='coerce').fillna(0)
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    return df.dropna(subset=['period']).assign(period=lambda d: d['period'].astype(int))


def build_adjacency():
    print("Loading Comtrade files...")
    raw = _load_raw()
    print(f"  Total rows loaded: {len(raw):,}")

    # ── Export records: reporter exported to partner ───────────────────────────
    exp = raw[raw['flow'].isin(['X', 'RX', 'DX'])].copy()
    exp = exp[exp['reporter'].isin(VALID_ISO) & exp['partner'].isin(VALID_ISO)]
    exp['from'] = exp['reporter'].map(NODE_MAP)
    exp['to']   = exp['partner'].map(NODE_MAP)
    exp = exp[exp['from'] != exp['to']]

    # ── Import records: reporter imported FROM partner → swap for export direction
    imp = raw[raw['flow'].isin(['M', 'FM', 'RM'])].copy()
    imp = imp[imp['reporter'].isin(VALID_ISO) & imp['partner'].isin(VALID_ISO)]
    imp['from'] = imp['partner'].map(NODE_MAP)   # partner = exporter
    imp['to']   = imp['reporter'].map(NODE_MAP)  # reporter = importer
    imp = imp[imp['from'] != imp['to']]

    agg_e = exp.groupby(['period','from','to'])['value'].sum().reset_index()
    agg_i = imp.groupby(['period','from','to'])['value'].sum().reset_index()

    # Prefer export records; fill gaps (e.g. SAU never reports) with import-derived
    m = agg_e.merge(agg_i, on=['period','from','to'], how='outer', suffixes=('_e','_i')).fillna(0)
    m['value'] = np.where(m['value_e'] > 0, m['value_e'], m['value_i'])
    trade = m[['period','from','to','value']]

    all_periods = sorted(trade['period'].unique())
    print(f"  Periods: {all_periods[0]} to {all_periods[-1]} ({len(all_periods)} months)")
    print(f"  Unique (from,to) pairs: {trade[['from','to']].drop_duplicates().shape[0]}")

    # ── Gap check ─────────────────────────────────────────────────────────────
    y0,m0 = all_periods[0]//100, all_periods[0]%100
    y1,m1 = all_periods[-1]//100, all_periods[-1]%100
    full, y,m = [], y0, m0
    while y*100+m <= y1*100+m1:
        full.append(y*100+m); m+=1
        if m>12: m,y=1,y+1
    gaps = sorted(set(full)-set(all_periods))
    print(f"  Gaps: {len(gaps)} missing months{(' → '+str(gaps[:6])) if gaps else ''}")

    # ── Build matrices ─────────────────────────────────────────────────────────
    parq_rows, read_rows = [], []
    for period in all_periods:
        sub = trade[trade['period']==period]
        mat = np.zeros((N_NODES, N_NODES))
        for _, row in sub.iterrows():
            mat[int(row['from']), int(row['to'])] += row['value']
        rs = mat.sum(axis=1, keepdims=True)
        norm = np.where(rs>0, mat/rs, 0.0)

        prow = {'period': period}
        for r in range(N_NODES):
            for c in range(N_NODES):
                prow[f'col_{r}_{c}'] = norm[r,c]
        parq_rows.append(prow)

        for i in range(N_NODES):
            for j in range(N_NODES):
                if i!=j:
                    read_rows.append({'period':period,'from_node':i,'to_node':j,
                        'from_name':NODE_NAMES[i],'to_name':NODE_NAMES[j],
                        'raw_value_usd':mat[i,j],'normalised_weight':norm[i,j]})

    df_p = pd.DataFrame(parq_rows)
    df_r = pd.DataFrame(read_rows)

    # ── Sample matrices ────────────────────────────────────────────────────────
    for sp in [201901, 202206, 202301]:
        row = df_p[df_p['period']==sp]
        if row.empty: print(f"\n{sp}: not in data"); continue
        mat = row.iloc[0][[f'col_{r}_{c}' for r in range(N_NODES) for c in range(N_NODES)]].values.reshape(N_NODES,N_NODES)
        ns  = [NODE_NAMES[i].split('/')[1] for i in range(N_NODES)]
        print(f"\nAdjacency {sp}:")
        print(f"  {'':>9}"+"".join(f"{n:>10}" for n in ns))
        for i in range(N_NODES):
            print(f"  {ns[i]:>9}"+"".join(f"{mat[i,j]:10.4f}" for j in range(N_NODES)))

    print("\nExport coverage:")
    for i in range(N_NODES):
        cols = [f'col_{i}_{j}' for j in range(N_NODES) if j!=i]
        present = (df_p[cols].sum(axis=1)>0).sum()
        print(f"  Node {i} {NODE_NAMES[i]}: {present}/{len(df_p)} periods")

    df_p.to_parquet(OUT_PARQ, index=False)
    df_r.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_PARQ}  shape={df_p.shape}")
    print(f"Saved: {OUT_CSV}  shape={df_r.shape}")
    return df_p


if __name__ == "__main__":
    build_adjacency()
