from transformers import BertForMaskedLM, BertTokenizerFast,Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from dataset import StreamingIRDataset  # Assuming this is your custom dataset class
from CustomTrainer import CustomTrainer

# Directory where the model was saved
model_directory = "./results/model"  # Replace with your actual directory path

# Load the model and tokenizer
model = BertForMaskedLM.from_pretrained("./results/model")
tokenizer = BertTokenizerFast(tokenizer_file="llvm_ir_wordpiece_tokenizer_replaced.json",
                              unk_token="[UNK]",
                              pad_token="[PAD]",
                              cls_token="[CLS]",
                              sep_token="[SEP]",
                              mask_token="[MASK]")

if "[NUM]" not in tokenizer.vocab:
    tokenizer.add_tokens(["[NUM]"])

# Define evaluation dataset and DataLoader
eval_dataset = StreamingIRDataset(batch_size=8, num_data=100, tokenizer=tokenizer, start_index=10000)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
eval_dataloader = DataLoader(eval_dataset, batch_size=None, collate_fn=data_collator, pin_memory=False)

# Define Trainer arguments (minimal configuration for evaluation)
training_args = TrainingArguments(
    output_dir="./results/eval",
    per_device_eval_batch_size=4,
)

# Initialize the Trainer with the loaded model
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    custom_eval_dataloader=eval_dataloader
)

# Run evaluation
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
