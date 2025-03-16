from transformers import BertConfig, BertForMaskedLM, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from CustomTrainer import CustomTrainer
from dataset import StreamingIRDataset

# Load your custom WordPiece tokenizer trained on LLVM IR
tokenizer = BertTokenizerFast(tokenizer_file="llvm_ir_wordpiece_tokenizer_replaced.json",
                              unk_token="[UNK]",
                              pad_token="[PAD]",
                              cls_token="[CLS]",
                              sep_token="[SEP]",
                              mask_token="[MASK]")

if "[NUM]" not in tokenizer.vocab:
    tokenizer.add_tokens(["[NUM]"])

# Step 1: Define a New BERT Model Configuration (starting from scratch, not pretrained)
config = BertConfig(
  vocab_size=tokenizer.vocab_size,
  hidden_size=768,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  max_position_embeddings=512,
)
model = BertForMaskedLM(config)
model.resize_token_embeddings(len(tokenizer))

# Step 3: Create DataLoader with the streaming dataset
streaming_train_dataset = StreamingIRDataset(batch_size=4, num_data=1000, tokenizer=tokenizer)
# TODO
streaming_eval_dataset = StreamingIRDataset(batch_size=4, num_data=100, split='train', tokenizer=tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_dataloader = DataLoader(streaming_train_dataset, batch_size=None, collate_fn=data_collator, pin_memory=False)
eval_dataloader = DataLoader(streaming_eval_dataset, batch_size=None, collate_fn=data_collator, pin_memory=False)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
  output_dir="./results",
  overwrite_output_dir=True,
  num_train_epochs=1, # set as 1 to iterate through the loop
  per_device_train_batch_size=4,
  save_steps=10_000,
  max_steps=500,
  save_total_limit=2,
  logging_dir='./logs',
)

# Step 5: Initialize Trainer with DataLoader instead of Dataset
trainer = CustomTrainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  # train_dataset=train_dataloader,  # Leave train_dataset empty since we're using DataLoader
  # train_dataloader=train_dataloader
  custom_train_dataloader=train_dataloader,
  custom_eval_dataloader=eval_dataloader
)

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    trainer.train()  # Run training for one "epoch"
    if epoch % 10 == 0:
      trainer.save_model(f"./results/checkpoint-epoch-{epoch + 1}")  # Save after each epoch
    eval_results = trainer.evaluate()  # Optionally evaluate after each epoch
    print(f"Evaluation results for epoch {epoch + 1}: {eval_results}")

# Step 6: Start Training
# trainer.train()

trainer.evaluate()
trainer.save_model("./results/model")
