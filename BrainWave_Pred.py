import numpy as np
from joblib import load
import os
# from __future__ import unicode_literals, print_function
from socket import socket, AF_INET, SOCK_DGRAM
from pythonosc import osc_message
import time

HOST = ''
PORT = 8001
class EEGPredictor:
    def __init__(self, model_dir='trained_model'):
        self.model, self.scaler = self._load_model(model_dir)
        self.class_names = {0: 'neutral', 1: 'right', 2: 'left'}
        self.feature_names = [
            'alpha', 'beta', 'theta', 'delta', 'gamma',
            'alpha_beta_ratio', 'theta_beta_ratio', 'alpha_theta_ratio',
            'alpha_rel', 'beta_rel', 'theta_rel', 'delta_rel', 'gamma_rel',
            'alpha_log', 'beta_log', 'theta_log', 'delta_log', 'gamma_log'
        ]

    def _load_model(self, model_dir):
        model_path = os.path.join(model_dir, 'rf_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        return load(model_path), load(scaler_path)

    def _create_features(self, basic_features):
        """基本的な脳波データから追加の特徴量を生成"""
        features = {}
        
        # 基本特徴量
        alpha, beta, theta, delta, gamma = basic_features
        features.update({
            'alpha': alpha, 'beta': beta, 'theta': theta,
            'delta': delta, 'gamma': gamma
        })
        
        # 比率の計算
        features['alpha_beta_ratio'] = alpha / beta if beta != 0 else 0
        features['theta_beta_ratio'] = theta / beta if beta != 0 else 0
        features['alpha_theta_ratio'] = alpha / theta if theta != 0 else 0
        
        # 相対パワーの計算
        total_power = sum(basic_features)
        if total_power != 0:
            features.update({
                'alpha_rel': alpha / total_power,
                'beta_rel': beta / total_power,
                'theta_rel': theta / total_power,
                'delta_rel': delta / total_power,
                'gamma_rel': gamma / total_power
            })
        else:
            features.update({
                'alpha_rel': 0, 'beta_rel': 0, 'theta_rel': 0,
                'delta_rel': 0, 'gamma_rel': 0
            })
        
        # 対数変換
        features.update({
            'alpha_log': np.log1p(alpha) if alpha > 0 else 0,
            'beta_log': np.log1p(beta) if beta > 0 else 0,
            'theta_log': np.log1p(theta) if theta > 0 else 0,
            'delta_log': np.log1p(delta) if delta > 0 else 0,
            'gamma_log': np.log1p(gamma) if gamma > 0 else 0
        })
        
        # 特徴量を正しい順序で並べ替え
        return [features[name] for name in self.feature_names]

    def predict(self, eeg_data):
        """脳波データから予測を行う"""
        # 特徴量の生成
        features = self._create_features(eeg_data)
        features = np.array(features).reshape(1, -1)
        
        # スケーリングと予測
        scaled_data = self.scaler.transform(features)
        prediction = self.model.predict(scaled_data)[0]
        probabilities = self.model.predict_proba(scaled_data)[0]
        
        # 結果の整形
        result = {
            'predicted_class': self.class_names[prediction],
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result
def process_eeg_data(raw_eeg_data):
    """
    脳波データを処理して予測を行う
    
    Parameters:
    raw_eeg_data: [alpha, beta, theta, delta, gamma]の形式の脳波データ
    """
    predictor = EEGPredictor()
    result = predictor.predict(raw_eeg_data)
    
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Class probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.2f}")
    
    return result

def Convert_BrainWave(data):
    msg = osc_message.OscMessage(data)
    #print(msg.params)
    types = msg.address
    arguments = []
    if  types == "/Attention":
        arguments.append("Attention")
        arguments.append(float(msg.params[0]))
        
    elif types == "/Meditation":
        arguments.append("Meditation")
        arguments.append(float(msg.params[0]))
        
    elif types == "/BandPower":
        arguments.append("BandPower")
        arguments += list(map(float, msg.params[0].split(";")))
        #print(map(float, msg.params[0].split(";")))
    return arguments
s = socket(AF_INET, SOCK_DGRAM)
s.bind((HOST, PORT))
while True:
    # 実際の脳波データをここに入れます
    print("受信待ち")
    data, address = s.recvfrom(1024)
    received_data = Convert_BrainWave(data=data)
    received_data = received_data[1:]
    print(received_data)
    result = process_eeg_data(received_data)
    time.sleep(0.5)
