import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

class EEGClassifier:
    def __init__(self, random_state: int = 42):
        """
        EEG信号の分類モデルを管理するクラス
        
        Parameters:
        random_state: int, 乱数シード
        """
        self.random_state = random_state
        self.models = self._define_models()
        self.best_models = {}
        self.feature_importance = {}
        
    def _define_models(self) -> Dict[str, Dict[str, Any]]:
        """モデルとハイパーパラメータの定義"""
        return {
            'svm': {
                'model': SVC(probability=True, random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def train_and_evaluate(self, processed_data: Dict[str, Any], cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        モデルの学習と評価を実行
        
        Parameters:
        processed_data: preprocess_eeg_dataからの出力
        cv_folds: クロスバリデーションの分割数
        
        Returns:
        Dict: 各モデルの評価結果
        """
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        feature_names = processed_data['feature_names']
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\nTraining {name}...")
            
            # クロスバリデーション設定
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # グリッドサーチ
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring=['accuracy', 'f1_macro', 'roc_auc_ovr'],
                refit='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            
            # モデルの学習
            grid_search.fit(X_train, y_train)
            
            # 最適モデルの保存
            best_model = grid_search.best_estimator_
            self.best_models[name] = best_model
            
            # テストデータでの予測
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            
            # 特徴量の重要度（可能な場合）
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # 評価指標の計算
            lb = LabelBinarizer()
            lb.fit(y_train)
            y_test_bin = lb.transform(y_test)
            
            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_results': {
                    metric: grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
                    for metric in ['accuracy', 'f1_macro', 'roc_auc_ovr']
                },
                'test_accuracy': grid_search.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
            }
            
            # 結果の表示
            self._print_results(name, results[name])
        
        return results
    
    def _print_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """結果の表示"""
        print(f"\nResults for {model_name}:")
        print("\nBest Parameters:")
        print(results['best_params'])
        print("\nCross-validation Results:")
        for metric, value in results['cv_results'].items():
            print(f"{metric}: {value:.4f}")
        print("\nTest Set Results:")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20, output_path: str = None) -> None:
        """
        特徴量の重要度をプロット
        
        Parameters:
        model_name: 表示するモデルの名前
        top_n: 表示する特徴量の数
        """
        if model_name not in self.feature_importance:
            print(f"Feature importance not available for {model_name}")
            return
            
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Important Features ({model_name})')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                plt.show()
    
    def plot_confusion_matrix(self, model_name: str, results: Dict[str, Dict[str, Any]], 
                            output_path: str = None) -> None:
        """
        混同行列をプロットする

        Parameters:
        model_name: str, モデルの名前
        results: Dict, モデルの評価結果
        output_path: str, 保存先のパス（Noneの場合は表示のみ）
        """
        if model_name not in results:
            print(f"Results not found for {model_name}")
            return
            
        cm = results[model_name]['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
        """
        混同行列をプロット
        
        Parameters:
        model_name: 表示するモデルの名前
        results: train_and_evaluateの結果
        """
        if model_name not in results:
            print(f"Results not found for {model_name}")
            return
            
        cm = results[model_name]['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

# 使用例
if __name__ == "__main__":
    # データの前処理（前回のコードで実行）
    # processed_data = preprocess_eeg_data("path_to_your_eeg_data.csv")
    
    # モデルのトレーニングと評価
    # classifier = EEGClassifier()
    # results = classifier.train_and_evaluate(processed_data)
    
    # 特徴量の重要度の表示
    # classifier.plot_feature_importance('random_forest')
    
    # 混同行列の表示
    # classifier.plot_confusion_matrix('random_forest', results)
    pass