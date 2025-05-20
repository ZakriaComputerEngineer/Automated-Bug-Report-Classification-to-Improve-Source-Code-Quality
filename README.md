# Automated-Bug-Report-Classification-to-Improve-Source-Code-Quality

This repository contains the code and resources for the research on automatically classifying software bug reports to predict their potential impact on source code quality, using a fine-tuned CodeBERT model. The goal is to identify "Code Quality Impacting Bugs" (CQIBs) versus "Non-Code-Quality-Impacting Bugs" (NQIBs) proactively.

## Abstract

Maintaining high source code quality is crucial. Bug fixes, while necessary, can inadvertently introduce code smells, degrading quality. This project proposes a CodeBERT-based model to automatically classify bug reports into CQIBs and NQIBs. We fine-tuned CodeBERT using a continual learning approach on a dataset comprising over 25,000 balanced bug report entries from four Apache projects (Geode, CloudStack, Camel, and HBase). Our model achieved a final accuracy of 91.21%, significantly outperforming prior baseline studies. These results highlight the superior capability of transformer models for this task.

## Project Structure

-   `codeBERT_exp2_1.ipynb`: Jupyter notebook containing the main implementation for data preprocessing, model fine-tuning (both continual and individual learning approaches), and evaluation.
-   `*.csv`: Dataset files (e.g., `Hbase_DE - v.01.csv`, etc.). *(Note: You might not want to push large CSVs to GitHub directly; consider using Git LFS or providing download links).*
-   `results/`: Directory where trained model checkpoints and evaluation outputs are saved by the notebook.
-   `images/` (Optional): If you have diagrams or plots you want to include in the README.
-   `README.md`: This file.

## Features

-   **Advanced Text Preprocessing:**
-   **CodeBERT Fine-Tuning:** Implementation for fine-tuning the `microsoft/codebert-base` model for binary sequence classification.
-   **Data Handling:**
    *   Loading data from multiple CSV files.
    *   Data cleaning and NaN value handling.
    *   Class balancing using upsampling.
    *   Splitting data into training and validation sets.
-   **Training Strategies:**
    1.  **Continual Learning:** Model is trained sequentially on accumulating datasets from different projects.
    2.  **Individual Learning:** Separate models are trained for each project dataset.
-   **Evaluation:** Uses standard metrics (Accuracy, F1-score, Precision, Recall) and generates confusion matrices.

## Requirements

-   Python 3.x
-   Pandas
-   NumPy
-   Scikit-learn
-   PyTorch
-   Hugging Face Transformers (`transformers`)
-   Hugging Face Datasets (`datasets`)
-   Hugging Face Evaluate (`evaluate`)
-   NLTK (for `punkt` and `stopwords`)
-   Matplotlib & Seaborn (for visualization)

You can install the main Python packages using pip:
```bash
pip install pandas numpy scikit-learn torch transformers datasets evaluate nltk matplotlib seaborn jupyter
```
Then, run the following in a Python interpreter to download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

1.  **Dataset:**
    *   Ensure your dataset CSV files (e.g., `Hbase_DE - v.01.csv`, `CloudStack_DE - v.01.csv`, `Camel_DE - v.02.csv`, `Geode_DE - v.01.csv`) are in the same directory as the notebook or update the `file_paths` list in the notebook.
    *   The CSVs should have two columns: `text` (bug report summary + description) and `label` (0 for NQIB, 1 for CQIB), with no header.
2.  **Run the Jupyter Notebook:**
    *   Open `codeBERT_exp2_1.ipynb` in Jupyter Lab or Jupyter Notebook.
    *   Execute the cells sequentially.
    *   The notebook handles data loading, preprocessing, balancing, model training (it's currently set up to train on the combined dataset from scratch as per cell 4, but you can adapt it for continual learning or individual models by modifying data loading and model checkpoint loading).
    *   
3.  **Model Saving:**
    *   The trained model and tokenizer will be saved to a directory (e.g., `./codebert-exp2-model_1` as per the last cell).
    *   Intermediate checkpoints during training are saved in `./results/`.

The notebook also generates confusion matrices and training/validation loss plots (if a new training run is performed). Specific results for continual learning stages and individual model trainings are detailed within the project's associated thesis/paper.

## Citation

If you use this work, please cite the associated thesis/paper:
```
[]
```

## License

This Research is based on masters Thesis of Nust Unibversity, use for academic research purposes only.]
```
