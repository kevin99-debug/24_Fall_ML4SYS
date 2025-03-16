import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling
from dataset import StreamingIRDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load your custom WordPiece tokenizer trained on LLVM IR
tokenizer = BertTokenizerFast(
    tokenizer_file="llvm_ir_wordpiece_tokenizer_replaced.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Ensure "[NUM]" token is in the vocabulary
if "[NUM]" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["[NUM]"])
    print("Added '[NUM]' token to the tokenizer.")

# Load the trained BertForMaskedLM model
model = BertForMaskedLM.from_pretrained("./results/model")
model.to(device)
model.eval()  # Set the model to evaluation mode
print("Loaded trained BertForMaskedLM model.")

# Create DataCollator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Load the evaluation dataset
streaming_eval_dataset = StreamingIRDataset(
    batch_size=4,
    num_data=1000,  # Adjust as needed
    split='train',  # Change to 'validation' or appropriate split if available
    tokenizer=tokenizer,
)
print("Loaded evaluation dataset.")

# Create DataLoader for evaluation
eval_dataloader = DataLoader(
    streaming_eval_dataset,
    batch_size=None,  # Streaming datasets handle batching internally
    collate_fn=data_collator,
    pin_memory=True if torch.cuda.is_available() else False
)
print("Created evaluation DataLoader.")

# Initialize variables for MLM loss computation
total_loss = 0.0
total_masked_tokens = 0

# Define the loss function (Cross-Entropy Loss is used internally by Hugging Face's BertForMaskedLM)
# No need to define it explicitly unless you want to customize it

# Disable gradient calculations for evaluation
with torch.no_grad():
    for batch_idx, batch in enumerate(eval_dataloader):
        input_ids = batch['input_ids'].to(device)      # Shape: (batch_size, seq_length)
        labels = batch['labels'].to(device)            # Shape: (batch_size, seq_length)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss  # This is the MLM loss (Cross-Entropy Loss) for the batch

        # Accumulate the total loss
        # To get the total loss over all masked tokens, multiply by the number of masked tokens in the batch
        # Hugging Face's loss is averaged over the number of masked tokens
        # Therefore, to get the sum, multiply by the batch's masked tokens
        # First, find the number of masked tokens in this batch
        masked_tokens = (labels != -100).sum().item()

        total_loss += loss.item() * masked_tokens  # Sum of losses
        total_masked_tokens += masked_tokens      # Total number of masked tokens

        # Optional: Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches.")
            average_mlm_loss = total_loss / total_masked_tokens
            print(f"Mean Masked Language Modeling (MLM) Loss on Evaluation Set: {average_mlm_loss:.4f}")

# Calculate Mean Masked Language Modeling (MLM) Loss
if total_masked_tokens > 0:
    average_mlm_loss = total_loss / total_masked_tokens
    print(f"\nMean Masked Language Modeling (MLM) Loss on Evaluation Set: {average_mlm_loss:.4f}")
else:
    print("No masked tokens found in the evaluation dataset.")
