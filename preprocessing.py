import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import List, Dict, Tuple, Union, Any
import logging

def validate_data(df: pd.DataFrame) -> Tuple[bool, str, Tuple[List[int], List[str]]]:
    """
    データの妥当性をチェック
    
    Parameters:
    df: DataFrame 入力データ
    
    Returns:
    bool: データが有効かどうか
    str: エラーメッセージ（エラーがある場合）
    tuple: (クリーニングが必要な行のインデックス, クリーニングの理由)
    """
    expected_columns = ['alpha', 'beta', 'theta', 'delta', 'gamma', 'key']
    valid_keys = [0, 1, 2]  # ニュートラル、右、左
    rows_to_clean = []
    cleaning_reasons = []
    
    # カラムの確認
    if not all(col in df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df.columns]
        return False, f"Missing columns: {missing_cols}", ([], [])
        
    # 数値データの確認
    numerical_cols = ['alpha', 'beta', 'theta', 'delta', 'gamma']
    for col in numerical_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            return False, f"Column {col} is not numeric", ([], [])
        
        # 負の値のチェック
        negative_mask = df[col] < 0
        if negative_mask.any():
            negative_rows = df[negative_mask].index.tolist()
            rows_to_clean.extend(negative_rows)
            cleaning_reasons.extend([f'Negative value in {col}'] * len(negative_rows))
        
        # NaNのチェック
        nan_mask = df[col].isna()
        if nan_mask.any():
            nan_rows = df[nan_mask].index.tolist()
            rows_to_clean.extend(nan_rows)
            cleaning_reasons.extend([f'NaN in {col}'] * len(nan_rows))

    # keyの値の確認
    key_nan_mask = df['key'].isna()
    if key_nan_mask.any():
        key_nan_rows = df[key_nan_mask].index.tolist()
        rows_to_clean.extend(key_nan_rows)
        cleaning_reasons.extend(['NaN in key'] * len(key_nan_rows))

    valid_rows = ~df['key'].isna()
    invalid_keys = [k for k in df[valid_rows]['key'].unique() if k not in valid_keys]
    if invalid_keys:
        invalid_key_mask = df['key'].isin(invalid_keys)
        invalid_key_rows = df[invalid_key_mask].index.tolist()
        rows_to_clean.extend(invalid_key_rows)
        cleaning_reasons.extend(['Invalid key value'] * len(invalid_key_rows))
    
    # 重複を除去
    if rows_to_clean:
        unique_indices = []
        unique_reasons = []
        seen = set()
        for idx, reason in zip(rows_to_clean, cleaning_reasons):
            if idx not in seen:
                unique_indices.append(idx)
                unique_reasons.append(reason)
                seen.add(idx)
        return True, "Data needs cleaning", (sorted(unique_indices), 
                                          [unique_reasons[unique_indices.index(i)] for i in sorted(unique_indices)])
    
    return True, "Data is valid", ([], [])

def load_task_segments(file_path: str) -> List[pd.DataFrame]:
    """
    空行で区切られたタスクデータを読み込む
    
    Parameters:
    file_path: CSVファイルのパス
    
    Returns:
    List[DataFrame]: タスクセグメントのリスト
    """
    df = pd.read_csv(file_path)
    empty_rows = df.index[df.isnull().all(axis=1)].tolist()
    
    task_segments = []
    start_idx = 0
    
    for end_idx in empty_rows:
        if end_idx > start_idx:
            segment = df.iloc[start_idx:end_idx].reset_index(drop=True)
            if not segment.empty:
                task_segments.append(segment)
        start_idx = end_idx + 1
    
    if start_idx < len(df):
        segment = df.iloc[start_idx:].reset_index(drop=True)
        if not segment.empty:
            task_segments.append(segment)
    
    return task_segments

def compute_spectral_features(data: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """スペクトル特徴量の計算"""
    features = {}
    
    # FFTの計算
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, 1/fs)[:n//2]
    power_spectrum = 2.0/n * np.abs(yf[0:n//2])
    
    # スペクトル特徴量
    features['spectral_mean'] = np.mean(power_spectrum)
    features['spectral_std'] = np.std(power_spectrum)
    features['spectral_skew'] = pd.Series(power_spectrum).skew()
    features['spectral_kurtosis'] = pd.Series(power_spectrum).kurtosis()
    
    # ピーク周波数
    peak_freq = xf[np.argmax(power_spectrum)]
    features['peak_frequency'] = peak_freq
    
    return features

def create_features(segment: pd.DataFrame) -> Dict[str, float]:
    """1タスク分のデータから特徴量を生成"""
    features = {}
    basic_features = ['alpha', 'beta', 'theta', 'delta', 'gamma']
    
    # 基本統計量
    for feature in basic_features:
        # 時系列データの統計量
        features[f'{feature}_mean'] = segment[feature].mean()
        features[f'{feature}_std'] = segment[feature].std()
        features[f'{feature}_var'] = segment[feature].var()
        features[f'{feature}_skew'] = segment[feature].skew()
        features[f'{feature}_kurtosis'] = segment[feature].kurtosis()
        
        # トレンドと変化率
        x = np.arange(len(segment))
        features[f'{feature}_trend'] = np.polyfit(x, segment[feature].values, 1)[0]
        features[f'{feature}_max_diff'] = segment[feature].diff().max()
        features[f'{feature}_min_diff'] = segment[feature].diff().min()
        
        # スペクトル特徴量
        spectral_features = compute_spectral_features(segment[feature].values)
        for spec_name, spec_value in spectral_features.items():
            features[f'{feature}_{spec_name}'] = spec_value
    
    # 波形間の比率
    ratios = {
        'alpha_beta_ratio': segment['alpha'] / segment['beta'],
        'theta_beta_ratio': segment['theta'] / segment['beta'],
        'alpha_theta_ratio': segment['alpha'] / segment['theta']
    }
    
    for ratio_name, ratio_data in ratios.items():
        features[f'{ratio_name}_mean'] = ratio_data.mean()
        features[f'{ratio_name}_std'] = ratio_data.std()
    
    # パワーの合計と相対パワー
    total_power = segment[basic_features].sum(axis=1)
    for feature in basic_features:
        rel_power = segment[feature] / total_power
        features[f'{feature}_rel_mean'] = rel_power.mean()
        features[f'{feature}_rel_std'] = rel_power.std()
    
    return features

def preprocess_eeg_data(file_path: str, test_size: float = 0.2, 
                       random_state: int = 42, validate: bool = True,
                       clean_data: bool = True) -> Dict[str, Any]:
    """
    EEGデータの前処理パイプライン
    
    Parameters:
    file_path: str, 入力CSVファイルのパス
    test_size: float, テストデータの割合
    random_state: int, 乱数シード
    validate: bool, データの妥当性チェックを行うかどうか
    clean_data: bool, 問題のあるデータを自動的に除去するかどうか
    
    Returns:
    Dict: 前処理済みのデータセット
    """
    # データの読み込みと妥当性チェック
    df = pd.read_csv(file_path)
    if validate:
        is_valid, message, (rows_to_clean, cleaning_reasons) = validate_data(df)
        
        if not is_valid:
            raise ValueError(f"Invalid data: {message}")
            
        if rows_to_clean and clean_data:
            logging.warning(f"Found {len(rows_to_clean)} rows that need cleaning:")
            for idx, (row, reason) in enumerate(zip(rows_to_clean, cleaning_reasons)):
                logging.warning(f"Row {row}: {reason}")
            
            # 問題のある行を削除
            df = df.drop(rows_to_clean).reset_index(drop=True)
            logging.info(f"Removed {len(rows_to_clean)} problematic rows. Remaining rows: {len(df)}")
            
        elif rows_to_clean and not clean_data:
            raise ValueError(f"Data contains {len(rows_to_clean)} problematic rows. Set clean_data=True to automatically remove them.")
    
    # タスクセグメントの読み込み
    segments = load_task_segments(file_path)
    
    X = []
    y = []
    
    # 各セグメントの処理
    for segment in segments:
        if not segment.empty:
            features = create_features(segment)
            label = segment['key'].iloc[0]  # セグメント内の最初のラベルを使用
            
            X.append(features)
            y.append(label)
    
    if X and y:  # データが存在する場合のみ処理
        X = pd.DataFrame(X)
        y = np.array(y)
        
        # 特徴量の前処理
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns,
            'scaler': scaler
        }
    
    return None  # データが存在しない場合