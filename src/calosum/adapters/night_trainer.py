from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalDatasetExporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def export(self, dataset: list[dict[str, Any]], filename: str) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        export_path = self.output_dir / filename
        with export_path.open("w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return str(export_path)

class NightTrainer:
    """
    Executa a Destilação Episódica (Neuroplasticidade) baseada em DSPy.
    
    Carrega o dataset gerado pelo `SleepModeConsolidator`, aplica otimização 
    (BootstrapFewShot) para encontrar os melhores exemplos (good/corrected) 
    e salva o artefato compilado para ser carregado no bootstrap do LeftHemisphere.
    LoRA é tratado como fallback/opcional futuro.
    """

    def __init__(self, model_name: str, dataset_path: Path, output_dir: Path):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def run_training_cycle(self) -> dict[str, Any]:
        if not self.dataset_path.exists():
            return {"status": "skipped", "reason": "No dataset found"}

        try:
            # Em uma implementação real com DSPy, inicializaríamos:
            # import dspy
            # from dspy.teleprompt import BootstrapFewShot
            # dspy.settings.configure(lm=dspy.LM(self.model_name))
            # ...
            
            logger.info(f"Loading DSPy dataset from {self.dataset_path}")
            
            dataset = []
            with self.dataset_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))

            good_examples = [item for item in dataset if item.get("category") in ("good", "corrected")]
            
            if not good_examples:
                logger.info("No high-quality examples found for few-shot compilation.")
                os.remove(self.dataset_path)
                return {"status": "skipped", "reason": "No valid examples"}

            logger.info(f"Compiling DSPy artifacts using {len(good_examples)} examples")
            
            # Simulando compilação de prompt
            compiled_artifact = {
                "model_name": self.model_name,
                "few_shot_examples": good_examples[:5], # top 5 melhores
                "compiled_at": "night_cycle",
                "optimizer": "BootstrapFewShot_Mock"
            }
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.output_dir / "compiled_prompt.json"
            
            with artifact_path.open("w", encoding="utf-8") as f:
                json.dump(compiled_artifact, f, indent=2, ensure_ascii=False)
                
            logger.info(f"DSPy artifact saved to {artifact_path}")
            
            # O arquivo é limpo após o aprendizado para não repetir na noite seguinte
            os.remove(self.dataset_path)
            
            return {
                "status": "success", 
                "examples_learned": len(good_examples),
                "artifact_path": str(artifact_path)
            }
            
        except Exception as e:
            logger.error(f"Night training failed: {e}")
            return {"status": "error", "reason": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = NightTrainer(
        model_name="Qwen/Qwen-3.5-9B-Instruct",
        dataset_path=Path(".calosum-runtime/nightly_data/dspy_dataset.jsonl"),
        output_dir=Path(".calosum-runtime/dspy_artifacts/latest")
    )
    result = trainer.run_training_cycle()
    print(result)