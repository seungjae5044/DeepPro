import os
import json
import shutil
import torch
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path


class ModelManager:
    """모델별 checkpoint 및 실험 결과 관리"""
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.global_best_file = self.base_dir / "global_best.json"
        
    def get_model_dir(self, model_name: str) -> Path:
        """모델별 디렉토리 경로 반환"""
        model_dir = self.base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    def get_experiment_dir(self, model_name: str, timestamp: str = None) -> Path:
        """실험별 디렉토리 경로 반환"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.get_model_dir(model_name)
        exp_dir = model_dir / timestamp
        exp_dir.mkdir(exist_ok=True)
        return exp_dir
    
    def save_model_checkpoint(self, model, model_name: str, accuracy: float, 
                            epoch: int, is_best: bool = False, 
                            timestamp: str = None, config: dict = None) -> str:
        """모델 체크포인트 저장"""
        exp_dir = self.get_experiment_dir(model_name, timestamp)
        
        # 모델 저장
        model.eval()
        example_input = torch.randn(1, 3, 48, 48)
        if torch.cuda.is_available():
            example_input = example_input.cuda()
        
        traced_script = torch.jit.trace(model, example_input)
        
        if is_best:
            model_path = exp_dir / "model_best.pt"
        else:
            model_path = exp_dir / f"model_epoch_{epoch}.pt"
        
        traced_script.save(str(model_path))
        
        # 메타데이터 저장
        metadata = {
            'model_name': model_name,
            'accuracy': accuracy,
            'epoch': epoch,
            'timestamp': timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
            'is_best': is_best,
            'config': config
        }
        
        metadata_path = exp_dir / f"metadata_{'best' if is_best else f'epoch_{epoch}'}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 모델별 best 업데이트
        if is_best:
            self._update_model_best(model_name, str(model_path), accuracy, metadata)
        
        return str(model_path)
    
    def _update_model_best(self, model_name: str, model_path: str, 
                          accuracy: float, metadata: dict):
        """모델별 best 모델 업데이트"""
        model_dir = self.get_model_dir(model_name)
        model_best_file = model_dir / "model_best_info.json"
        
        # 현재 모델의 best 정보 로드
        current_best = None
        if model_best_file.exists():
            with open(model_best_file, 'r') as f:
                current_best = json.load(f)
        
        # 새로운 best 모델인지 확인
        if current_best is None or accuracy > current_best.get('accuracy', 0):
            # 모델별 best 파일 복사
            best_model_path = model_dir / "model_best.pt"
            shutil.copy2(model_path, best_model_path)
            
            # 모델별 best 정보 저장
            best_info = {
                'model_name': model_name,
                'accuracy': accuracy,
                'model_path': str(best_model_path),
                'source_path': model_path,
                'metadata': metadata,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(model_best_file, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            # 글로벌 best 업데이트 확인
            self._check_global_best(model_name, accuracy, str(best_model_path), best_info)
    
    def _check_global_best(self, model_name: str, accuracy: float, 
                          model_path: str, model_info: dict):
        """글로벌 best 모델 업데이트"""
        current_global_best = None
        
        if self.global_best_file.exists():
            with open(self.global_best_file, 'r') as f:
                current_global_best = json.load(f)
        
        # 새로운 글로벌 best인지 확인
        if current_global_best is None or accuracy > current_global_best.get('accuracy', 0):
            # checkpoints 디렉토리에 글로벌 best 복사
            checkpoints_dir = Path("./checkpoints")
            checkpoints_dir.mkdir(exist_ok=True)
            global_best_path = checkpoints_dir / "model_best.pt"
            
            shutil.copy2(model_path, global_best_path)
            
            # 글로벌 best 정보 저장
            global_best_info = {
                'model_name': model_name,
                'accuracy': accuracy,
                'model_path': str(global_best_path),
                'source_path': model_path,
                'model_info': model_info,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.global_best_file, 'w') as f:
                json.dump(global_best_info, f, indent=2)
            
            print(f"🎉 New global best model! {model_name} with {accuracy:.3f}% accuracy")
    
    def get_model_best_info(self, model_name: str) -> Optional[Dict]:
        """모델별 best 정보 조회"""
        model_dir = self.get_model_dir(model_name)
        model_best_file = model_dir / "model_best_info.json"
        
        if model_best_file.exists():
            with open(model_best_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_global_best_info(self) -> Optional[Dict]:
        """글로벌 best 정보 조회"""
        if self.global_best_file.exists():
            with open(self.global_best_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_all_models(self) -> Dict[str, Dict]:
        """모든 모델의 best 정보 조회"""
        models_info = {}
        
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir() and model_dir.name != "global_best.json":
                model_info = self.get_model_best_info(model_dir.name)
                if model_info:
                    models_info[model_dir.name] = model_info
        
        return models_info
    
    def cleanup_old_experiments(self, model_name: str, keep_last_n: int = 5):
        """오래된 실험 결과 정리 (최근 N개만 유지)"""
        model_dir = self.get_model_dir(model_name)
        
        # 타임스탬프 디렉토리들 찾기
        timestamp_dirs = [d for d in model_dir.iterdir() 
                         if d.is_dir() and d.name != "model_best.pt"]
        
        # 최신 순으로 정렬
        timestamp_dirs.sort(key=lambda x: x.name, reverse=True)
        
        # 오래된 것들 삭제
        for old_dir in timestamp_dirs[keep_last_n:]:
            shutil.rmtree(old_dir)
            print(f"Cleaned up old experiment: {old_dir}")