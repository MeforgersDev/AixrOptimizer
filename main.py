import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from huggingface_hub import HfApi
import aixr_optimizer as AixrOptimizer
import model_downloader as md

class AixrOptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Aixr Optimizer')

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.model_label = QLabel('Select Model:', self)
        left_layout.addWidget(self.model_label)

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])
        left_layout.addWidget(self.model_combo)

        self.tokenizer_label = QLabel('Select Tokenizer:', self)
        left_layout.addWidget(self.tokenizer_label)

        self.tokenizer_combo = QComboBox(self)
        self.tokenizer_combo.addItems(["bert-base-uncased", "roberta-base", "distilbert-base-uncased"])
        left_layout.addWidget(self.tokenizer_combo)

        self.device_label = QLabel('Select Device:', self)
        left_layout.addWidget(self.device_label)

        self.device_combo = QComboBox(self)
        self.device_combo.addItems(["CPU", "GPU"])
        left_layout.addWidget(self.device_combo)

        self.download_button = QPushButton('Download Model', self)
        self.download_button.clicked.connect(self.download_model)
        left_layout.addWidget(self.download_button)

        self.load_data_button = QPushButton('Load Dataset', self)
        self.load_data_button.clicked.connect(self.load_dataset)
        left_layout.addWidget(self.load_data_button)

        self.start_button = QPushButton('Start Training', self)
        self.start_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Training', self)
        left_layout.addWidget(self.stop_button)

        self.upload_button = QPushButton('Upload Model to HuggingFace', self)
        self.upload_button.clicked.connect(self.upload_model)
        left_layout.addWidget(self.upload_button)

        main_layout.addLayout(left_layout)

        self.dataset_preview = QTableWidget()
        main_layout.addWidget(self.dataset_preview)

        self.setLayout(main_layout)
        self.show()

    def download_model(self):
        model_name = self.model_combo.currentText()
        md.download_model(model_name)

    def load_dataset(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            df = pd.read_csv(file_name)
            self.show_dataset_preview(df)
            self.dataset_path = file_name

    def show_dataset_preview(self, df):
        self.dataset_preview.setRowCount(df.shape[0])
        self.dataset_preview.setColumnCount(df.shape[1])
        self.dataset_preview.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.dataset_preview.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def start_training(self):
        model_name = self.model_combo.currentText()
        tokenizer_name = self.tokenizer_combo.currentText()
        device = torch.device("cuda" if torch.cuda.is_available() and self.device_combo.currentText().lower() == 'gpu' else "cpu")
        model, tokenizer = md.load_model(model_name, device)
        train_dataset = md.load_dataset(self.dataset_path)  # replace with your dataset loading logic
        optimizer = AixrOptimizer(model.parameters())
        train_model(model, tokenizer, train_dataset, device, optimizer)

    def upload_model(self):
        model_name = self.model_combo.currentText()
        repo_name = model_name.replace("/", "_")
        api = HfApi()
        api.upload_folder(
            folder_path=f"./models/{model_name}",
            path_in_repo=".",
            repo_id=repo_name,
            repo_type="model"
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AixrOptimizerApp()
    sys.exit(app.exec_())

from transformers import Trainer, TrainingArguments

def train_model(model, tokenizer, train_dataset, device, optimizer):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None)
    )
    trainer.train()
