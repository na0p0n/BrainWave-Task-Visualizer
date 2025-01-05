import argparse
import os
import json
from datetime import datetime
import logging
from preprocessing import preprocess_eeg_data
from model_training import EEGClassifier

def setup_logging(output_dir):
    """ロギングの設定"""
    log_file = os.path.join(output_dir, f'eeg_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_results(results, output_dir):
    """結果の保存"""
    # 結果からモデルオブジェクトを除去（JSON化できないため）
    serializable_results = {}
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            k: v for k, v in model_results.items()
            if k not in ['model', 'confusion_matrix']
        }
        # confusion_matrixをリストに変換
        serializable_results[model_name]['confusion_matrix'] = \
            model_results['confusion_matrix'].tolist()

    # 結果をJSONファイルとして保存
    results_file = os.path.join(output_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='EEG信号分類プログラム')
    parser.add_argument('input_file', help='入力CSVファイルのパス')
    parser.add_argument('--output_dir', default='output', help='出力ディレクトリ')
    parser.add_argument('--test_size', type=float, default=0.2, help='テストデータの割合')
    parser.add_argument('--random_state', type=int, default=42, help='乱数シード')
    parser.add_argument('--cv_folds', type=int, default=5, help='クロスバリデーションの分割数')
    parser.add_argument('--skip_plots', action='store_true', help='プロットの生成をスキップ')
    
    args = parser.parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ロギングの設定
    setup_logging(args.output_dir)
    
    try:
        # 処理開始のログ
        logging.info(f"Processing started for file: {args.input_file}")
        
        # データの前処理
        logging.info("Starting data preprocessing...")
        processed_data = preprocess_eeg_data(
            args.input_file,
            test_size=args.test_size,
            random_state=args.random_state,
            validate=True
        )
        logging.info("Data preprocessing completed")

        # モデルのトレーニングと評価
        logging.info("Starting model training and evaluation...")
        classifier = EEGClassifier(random_state=args.random_state)
        results = classifier.train_and_evaluate(processed_data, cv_folds=args.cv_folds)
        logging.info("Model training and evaluation completed")

        # 結果の保存
        logging.info("Saving results...")
        save_results(results, args.output_dir)

        if not args.skip_plots:
            logging.info("Generating plots...")
            # 特徴量重要度のプロット
            for model_name in ['random_forest', 'xgboost']:
                classifier.plot_feature_importance(
                    model_name=model_name,
                    output_path=os.path.join(args.output_dir, f'feature_importance_{model_name}.png')
                )
                
            # 混同行列のプロット
            for model_name in results.keys():
                classifier.plot_confusion_matrix(
                    model_name=model_name,
                    results=results,
                    output_path=os.path.join(args.output_dir, f'confusion_matrix_{model_name}.png')
                )
            logging.info("Plot generation completed")

        logging.info("All processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()