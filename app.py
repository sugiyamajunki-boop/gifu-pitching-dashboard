import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import japanize_matplotlib
import os

# Streamlitのページ設定
st.set_page_config(page_title="岐阜第一高校 投手データ分析", layout="wide", initial_sidebar_state="expanded")

# 球種名の統一とカラーマッピング
PITCH_NAME_MAPPING = {
    '4_seam_fastball': 'ストレート',
    'fastball': 'ストレート',
    'two_seam_fastball': 'ツーシーム',
    'sinker': 'シンカー',
    'slider': 'スライダー',
    'sweeper': 'スイーパー',
    'slurve': 'スラーブ',
    'curveball': 'カーブ',
    'cutter': 'カットボール',
    'changeup': 'チェンジアップ',
    'splitter': 'スプリット'
}

COLOR_MAP = {
    'ストレート': '#e63946',
    'ツーシーム': '#a67c00',
    'シンカー': '#00a3e0',
    'スライダー': '#ff7a18',
    'スイーパー': '#ff5a00',
    'スラーブ': '#cc4400',
    'カーブ': '#32b141',
    'カットボール': '#9e2a2b',
    'チェンジアップ': '#ffcc00',
    'スプリット': '#7b2cbf',
    'フォーク': '#7b2cbf'
}

# データの読み込み関数
@st.cache_data
def load_data():
    # ※ GitHub等で公開する際は、同じディレクトリのCSVを読み込むようにパスを変更します
    # ここでは仮のパスとして設定
    game_path = '全試合統合_処理済データ.csv'
    tracking_path = '全トラッキング統合_処理済データ.csv'
    
    df_game, df_track = pd.DataFrame(), pd.DataFrame()
    
    if os.path.exists(game_path):
        df_game = pd.read_csv(game_path, encoding='utf-8-sig')
    
    if os.path.exists(tracking_path):
        df_track = pd.read_csv(tracking_path, encoding='utf-8-sig')
        # SmartScoutの球種名を日本語に統一
        if 'Pitch Type' in df_track.columns:
            df_track['球種'] = df_track['Pitch Type'].map(PITCH_NAME_MAPPING).fillna(df_track['Pitch Type'])
        # 不要なデータを削除
        df_track = df_track[~df_track['球種'].isin(['none', 'None', '', np.nan])]
        
    return df_game, df_track

# ==========================================
# メイン画面構築
# ==========================================
st.title("⚾ 岐阜第一高校野球部 投手データポータル")

df_game, df_track = load_data()

if df_game.empty and df_track.empty:
    st.warning("データファイルが見つかりません。'全試合統合_処理済データ.csv' と '全トラッキング統合_処理済データ.csv' を同じフォルダに配置してください。")
    st.stop()

# 投手のリストを取得
pitchers = []
if not df_game.empty and '投手' in df_game.columns:
    pitchers.extend(df_game['投手'].dropna().unique())
if not df_track.empty and 'Player' in df_track.columns:
    pitchers.extend(df_track['Player'].dropna().unique())
pitchers = sorted(list(set(pitchers)))

# サイドバーで投手を選択
st.sidebar.header("選手選択")
selected_pitcher = st.sidebar.selectbox("分析する投手を選んでください", pitchers, index=pitchers.index('山中 雄太') if '山中 雄太' in pitchers else 0)

# 選択した投手のデータを抽出
df_g = df_game[df_game['投手'] == selected_pitcher].copy() if not df_game.empty else pd.DataFrame()
df_t = df_track[df_track['Player'] == selected_pitcher].copy() if not df_track.empty else pd.DataFrame()

# タブの作成
tab1, tab2, tab3 = st.tabs(["📊 総合スタッツ・球種別成績", "⚾ 実戦配球・シークエンス", "📈 トラッキング・月別推移"])

# ------------------------------------------
# タブ1: 総合スタッツ・球種別成績 (Pixel Score)
# ------------------------------------------
with tab1:
    st.header("シーズン総合スタッツ")
    if not df_g.empty:
        # 簡易的なスタッツ計算 (防除率などはイニング計算が必要なため、概要のみ表示)
        total_pitches = len(df_g)
        strike_pitches = df_g['投球結果'].isin(['見逃し', '空振り', 'ファウル', 'インプレー']).sum()
        strike_rate = strike_pitches / total_pitches if total_pitches > 0 else 0
        
        pa_data = df_g[df_g['打席'] == 1]
        so = pa_data['三振四球'].isin(['空振り三振', '見逃し三振']).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総投球数", f"{total_pitches} 球")
        col2.metric("ストライク率", f"{strike_rate*100:.1f} %")
        col3.metric("奪三振", f"{so} 個")
        
        st.subheader("球種別パフォーマンス")
        # 球種別の集計
        pitch_stats = []
        for pitch_type, group in df_g.groupby('球種'):
            if pd.isna(pitch_type) or pitch_type == '不明': continue
            count = len(group)
            if count < 5: continue # 5球未満は除外
            
            usage = count / total_pitches
            avg_velo = pd.to_numeric(group['球速'], errors='coerce').mean()
            
            s_pitches = group['投球結果'].isin(['見逃し', '空振り', 'ファウル', 'インプレー']).sum()
            s_rate = s_pitches / count if count > 0 else 0
            
            swing = group['投球結果'].isin(['空振り', 'ファウル', 'インプレー']).sum()
            whiff = (group['投球結果'] == '空振り').sum()
            whiff_rate = whiff / swing if swing > 0 else 0
            
            pitch_stats.append({
                '球種': pitch_type,
                '投球数': count,
                '使用率(%)': round(usage * 100, 1),
                '平均球速(km/h)': round(avg_velo, 1) if pd.notna(avg_velo) else '-',
                'ストライク率(%)': round(s_rate * 100, 1),
                'Whiff%(空振り率)': round(whiff_rate * 100, 1)
            })
            
        if pitch_stats:
            st.dataframe(pd.DataFrame(pitch_stats).set_index('球種'), use_container_width=True)
    else:
        st.write("Pixel Scoreのデータがありません。")

# ------------------------------------------
# タブ2: 実戦配球・シークエンス (MoonOMT風)
# ------------------------------------------
with tab2:
    st.header("カウント別球種分布")
    if not df_g.empty:
        # しきい値で球種を絞る
        threshold = 0.05
        pitch_counts = df_g['球種'].value_counts(normalize=True)
        valid_pitches = pitch_counts[pitch_counts >= threshold].index.tolist()
        df_valid_seq = df_g[df_g['球種'].isin(valid_pitches)].copy()
        
        df_valid_seq['ボール'] = pd.to_numeric(df_valid_seq['ボール'], errors='coerce')
        df_valid_seq['ストライク'] = pd.to_numeric(df_valid_seq['ストライク'], errors='coerce')
        df_valid_seq = df_valid_seq.dropna(subset=['ボール', 'ストライク', '球種'])
        
        df_valid_seq['count_str'] = df_valid_seq['ボール'].astype(int).astype(str) + "-" + df_valid_seq['ストライク'].astype(int).astype(str)
        count_order = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']
        
        count_dist = df_valid_seq.groupby(['count_str', '球種']).size().unstack(fill_value=0)
        count_dist = count_dist.reindex(count_order).fillna(0)
        count_dist_pct = count_dist.div(count_dist.sum(axis=1).replace(0, 1), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#15161a')
        bottom = np.zeros(len(count_order))
        for pitch in count_dist_pct.columns:
            if count_dist_pct[pitch].sum() > 0:
                color = COLOR_MAP.get(pitch, '#888888')
                values = count_dist_pct[pitch].values
                ax.bar(count_order, values, bottom=bottom, label=pitch, color=color, edgecolor='#15161a')
                bottom += values
                
        ax.set_facecolor('#15161a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=6, frameon=False, labelcolor='white')
        
        st.pyplot(fig)
        
    else:
        st.write("配球データがありません。")

# ------------------------------------------
# タブ3: トラッキング・月別推移 (SmartScout)
# ------------------------------------------
with tab3:
    st.header("ピッチデザイン (変化量 × 球速帯)")
    if not df_t.empty:
        col_hb = 'Horizontal Break (cm)'
        col_vb = 'Vertical Break (cm)'
        col_velo = 'Initial Velocity (km/h)'
        col_spin = 'Total Spin (RPM)'
        
        for col in [col_hb, col_vb, col_velo, col_spin]:
            if col in df_t.columns:
                df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
        
        fig2, (ax_hb, ax_spin) = plt.subplots(1, 2, figsize=(12, 5))
        
        if col_hb in df_t.columns and col_vb in df_t.columns:
            sns.scatterplot(data=df_t, x=col_hb, y=col_vb, hue='球種', palette=COLOR_MAP, s=80, alpha=0.7, ax=ax_hb)
            ax_hb.axhline(0, color='gray', linestyle='--')
            ax_hb.axvline(0, color='gray', linestyle='--')
            ax_hb.set_title("横変化 × 縦変化")
        
        if col_velo in df_t.columns and col_spin in df_t.columns:
            sns.scatterplot(data=df_t, x=col_velo, y=col_spin, hue='球種', palette=COLOR_MAP, s=80, alpha=0.7, ax=ax_spin, legend=False)
            ax_spin.set_title("球速 × 回転数")
            
        st.pyplot(fig2)
        
        st.subheader("月別トラッキング推移")
        # 日付処理
        if 'Date' in df_t.columns:
            df_t['Date'] = pd.to_datetime(df_t['Date'], errors='coerce')
            df_t['年月'] = df_t['Date'].dt.strftime('%Y-%m')
        elif 'Title' in df_t.columns:
            df_t['年月'] = df_t['Title'].astype(str).str.extract(r'(20\d{2}[-/]?\d{2})')[0]
            df_t['年月'] = df_t['年月'].str.replace('/', '-').str.replace(r'^(20\d{2})(\d{2})$', r'\1-\2', regex=True)
            
        if '年月' in df_t.columns:
            df_t = df_t.dropna(subset=['年月'])
            if not df_t.empty:
                monthly_stats = df_t.groupby(['年月', '球種']).agg({
                    col_velo: 'mean', col_spin: 'mean', col_vb: 'mean', col_hb: 'mean'
                }).reset_index()
                
                monthly_counts = df_t.groupby(['年月', '球種']).size().reset_index(name='投球数')
                monthly_stats = pd.merge(monthly_stats, monthly_counts, on=['年月', '球種'])
                monthly_stats = monthly_stats[monthly_stats['投球数'] >= 3] # 月3球未満は除外
                
                # 見やすく整形
                st.dataframe(monthly_stats.round(1), use_container_width=True)
    else:
        st.write("SmartScoutのデータがありません。")

