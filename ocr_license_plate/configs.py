import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("ocr_license_plate/Models/license_plate_ocr", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -."
        self.height = 32
        self.width = 200
        self.max_text_length = 15
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.train_epochs = 150
        #self.train_workers = 4