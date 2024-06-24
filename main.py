import sys
import os
import requests
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QLineEdit, 
                             QTextEdit, QProgressBar, QSpinBox, QFormLayout, QDoubleSpinBox)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
from huggingface_hub import HfApi, HfFolder
import aixr_optimizer as AixrOptimizer
import model_downloader as md
from transformers import Trainer, TrainingArguments

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
        self.populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self.update_tokenizer_and_config)
        left_layout.addWidget(self.model_combo)

        self.tokenizer_label = QLabel('Selected Tokenizer:', self)
        left_layout.addWidget(self.tokenizer_label)

        self.tokenizer_combo = QLabel('', self)
        left_layout.addWidget(self.tokenizer_combo)

        self.config_label = QLabel('Selected Config:', self)
        left_layout.addWidget(self.config_label)

        self.config_display = QLabel('', self)
        left_layout.addWidget(self.config_display)

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

        self.training_params_form = QFormLayout()
        
        self.num_epochs_input = QSpinBox(self)
        self.num_epochs_input.setRange(1, 100)
        self.num_epochs_input.setValue(3)
        self.training_params_form.addRow('Number of Epochs:', self.num_epochs_input)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setRange(1, 1024)
        self.batch_size_input.setValue(16)
        self.training_params_form.addRow('Batch Size:', self.batch_size_input)

        self.learning_rate_input = QDoubleSpinBox(self)
        self.learning_rate_input.setRange(1e-6, 1e-1)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setValue(1e-3)
        self.training_params_form.addRow('Learning Rate:', self.learning_rate_input)

        left_layout.addLayout(self.training_params_form)

        self.start_button = QPushButton('Start Training', self)
        self.start_button.clicked.connect(self.start_training)
        left_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Training', self)
        left_layout.addWidget(self.stop_button)

        self.repo_label = QLabel('Hugging Face Repo:', self)
        left_layout.addWidget(self.repo_label)

        self.repo_input = QLineEdit(self)
        left_layout.addWidget(self.repo_input)

        self.username_label = QLabel('Hugging Face Username:', self)
        left_layout.addWidget(self.username_label)

        self.username_input = QLineEdit(self)
        left_layout.addWidget(self.username_input)

        self.token_label = QLabel('Hugging Face Token:', self)
        left_layout.addWidget(self.token_label)

        self.token_input = QLineEdit(self)
        self.token_input.setEchoMode(QLineEdit.Password)
        left_layout.addWidget(self.token_input)

        self.upload_button = QPushButton('Upload Model to HuggingFace', self)
        self.upload_button.clicked.connect(self.upload_model)
        left_layout.addWidget(self.upload_button)

        self.progress_bar = QProgressBar(self)
        left_layout.addWidget(self.progress_bar)

        self.terminal_output = QTextEdit(self)
        self.terminal_output.setReadOnly(True)
        left_layout.addWidget(self.terminal_output)

        main_layout.addLayout(left_layout)

        self.dataset_preview = QTableWidget()
        main_layout.addWidget(self.dataset_preview)

        self.setLayout(main_layout)
        self.show()

    def populate_model_combo(self):
        try:
            api = HfApi()
            models = api.list_models()
            model_names = [model.modelId for model in models]
            self.model_combo.addItems(model_names)
        except Exception as e:
            self.log_message(f"Error fetching models: {str(e)}")

    def update_tokenizer_and_config(self):
        model_name = self.model_combo.currentText()
        try:
            config = AutoConfig.from_pretrained(model_name)
            self.config_display.setText(str(config))
            self.tokenizer_combo.setText(model_name)
        except Exception as e:
            self.log_message(f"Error fetching config or tokenizer: {str(e)}")

    def log_message(self, message):
        self.terminal_output.append(message)
        QApplication.processEvents()

    def download_model(self):
        model_name = self.model_combo.currentText()
        self.log_message(f"Downloading model: {model_name}")
        try:
            md.download_model(model_name)
            self.log_message(f"Model {model_name} downloaded successfully.")
        except Exception as e:
            self.log_message(f"Error downloading model: {str(e)}")

    def load_dataset(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            df = pd.read_csv(file_name)
            self.show_dataset_preview(df)
            self.dataset_path = file_name
            self.log_message(f"Dataset loaded: {file_name}")

    def show_dataset_preview(self, df):
        self.dataset_preview.setRowCount(df.shape[0])
        self.dataset_preview.setColumnCount(df.shape[1])
        self.dataset_preview.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.dataset_preview.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

    def start_training(self):
        model_name = self.model_combo.currentText()
        device = torch.device("cuda" if torch.cuda.is_available() and self.device_combo.currentText().lower() == 'gpu' else "cpu")
        num_epochs = self.num_epochs_input.value()
        batch_size = self.batch_size_input.value()
        learning_rate = self.learning_rate_input.value()
        try:
            model, tokenizer = md.load_model(model_name, device)
            train_dataset = md.load_dataset(self.dataset_path)
            optimizer = AixrOptimizer.AixrOptimizer(model.parameters(), lr=learning_rate)
            self.log_message("Training started...")
            self.train_model(model, tokenizer, train_dataset, device, optimizer, num_epochs, batch_size)
            self.log_message("Training completed successfully.")
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")

    def train_model(self, model, tokenizer, train_dataset, device, optimizer, num_epochs, batch_size):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
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

    def upload_model(self):
        model_name = self.model_combo.currentText()
        repo_name = self.repo_input.text()
        username = self.username_input.text()
        token = self.token_input.text()
        if not repo_name or not username or not token:
            self.log_message("Please provide the repository name, username, and token.")
            return
        self.log_message(f"Uploading model {model_name} to repository {repo_name}...")
        try:
            HfFolder.save_token(token)
            api = HfApi()
            api.upload_folder(
                folder_path=f"./models/{model_name}",
                path_in_repo=".",
                repo_id=f"{username}/{repo_name}",
                repo_type="model"
            )
            self.log_message(f"Model {model_name} uploaded to {repo_name} successfully.")
        except Exception as e:
            self.log_message(f"Error uploading model: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AixrOptimizerApp()
    sys.exit(app.exec_())
