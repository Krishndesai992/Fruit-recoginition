# Contributing Guidelines

Thank you for your interest in contributing to the **Fruit Recognition using Support Vector Machine based on Deep Features** repository!

## Workflow
1. **Fork** the repository and create a descriptive feature branch (`feature/your-feature-name`).
2. **Set up** the environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Format & Quality:** Follow PEP8 standards and keep code modular and documented.
4. **Testing:**
   - Verify dataset loading: `python src/data_loader.py`
   - Test feature extraction: `python src/feature_extractor.py --backbone resnet50`
   - Test SVM training & evaluation: `python src/train_svm.py` and `python src/evaluate.py`
   - Test inference: `python src/predict.py --image <path>`
5. **Pull Requests:** Submit a pull request with a clear description of changes, motivation, and verification steps.
