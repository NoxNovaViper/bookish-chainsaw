import os
import torch
from AI import NanoTransformer

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
dummy_model = NanoTransformer()

for model_name in ['architect', 'coder', 'validator', 'analyzer', 'fixer', 'refactorer', 'hypothesizer', 'synthesizer', 'checker']:
    torch.save(dummy_model.state_dict(), os.path.join(models_dir, f'{model_name}.pt'))

print("✅ Dummy models created successfully.")