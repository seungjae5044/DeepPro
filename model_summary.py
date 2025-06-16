#!/usr/bin/env python3
"""
모델별 실험 결과 요약 스크립트
"""

import json
from pathlib import Path
from utils.model_manager import ModelManager


def main():
    """모든 모델의 실험 결과 요약 출력"""
    model_manager = ModelManager()
    
    print("=" * 60)
    print("                    MODEL SUMMARY")
    print("=" * 60)
    
    # 전체 모델 정보 조회
    all_models = model_manager.list_all_models()
    
    if not all_models:
        print("No trained models found.")
        return
    
    # 모델별 정보 출력
    models_by_accuracy = sorted(all_models.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
    
    print(f"\n{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'Updated':<20}")
    print("-" * 60)
    
    for rank, (model_name, info) in enumerate(models_by_accuracy, 1):
        print(f"{rank:<4} {model_name:<20} {info['accuracy']:.3f}%{'':<3} "
              f"{info['updated_at'][:19].replace('T', ' ')}")
    
    # 글로벌 best 정보
    global_best = model_manager.get_global_best_info()
    if global_best:
        print("\n" + "=" * 60)
        print("                   GLOBAL BEST MODEL")
        print("=" * 60)
        print(f"Model: {global_best['model_name']}")
        print(f"Accuracy: {global_best['accuracy']:.3f}%")
        print(f"Updated: {global_best['updated_at'][:19].replace('T', ' ')}")
        print(f"Path: {global_best['model_path']}")
    
    # 실험 디렉토리 구조 안내
    print("\n" + "=" * 60)
    print("                 DIRECTORY STRUCTURE")
    print("=" * 60)
    print("experiments/")
    print("├── <model_name>/")
    print("│   ├── model_best.pt              # 해당 모델의 best checkpoint")
    print("│   ├── model_best_info.json       # best 모델 메타데이터")
    print("│   └── <timestamp>/               # 실험별 결과")
    print("│       ├── model_best.pt          # 해당 실험의 best checkpoint")
    print("│       └── metadata_best.json     # 상세 메타데이터")
    print("├── global_best.json               # 전체 최고 모델 정보")
    print("└── checkpoints/")
    print("    └── model_best.pt              # 전체 최고 모델 (글로벌 best)")


if __name__ == "__main__":
    main()