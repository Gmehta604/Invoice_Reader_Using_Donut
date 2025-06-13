from transformers import DonutProcessor, VisionEncoderDecoderModel
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch.optim import AdamW
from invoice_dataset import InvoiceDataset

# === CONFIG ===
image_root = "data/raw/images"
train_json = "data/processed/train.json"
val_json = "data/processed/val.json"
pretrained_ckpt = "naver-clova-ix/donut-base"

# === LOAD MODEL & PROCESSOR ===
processor = DonutProcessor.from_pretrained(pretrained_ckpt)
model = VisionEncoderDecoderModel.from_pretrained(pretrained_ckpt)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# === DATASETS ===
train_dataset = InvoiceDataset(train_json, image_root, processor)
val_dataset = InvoiceDataset(val_json, image_root, processor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# === PL WRAPPER ===
class DonutPLModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values=None, decoder_input_ids=None, labels=None, **kwargs):
        return self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("val_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

# === TRAIN ===
checkpoint_cb = ModelCheckpoint(dirpath="checkpoints", save_top_k=1, monitor="val_loss", mode="min")
logger = CSVLogger("logs", name="donut-invoice")

trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision=16,
    callbacks=[checkpoint_cb],
    logger=logger
)

pl_model = DonutPLModule(model)
trainer.fit(pl_model, train_loader, val_loader)
