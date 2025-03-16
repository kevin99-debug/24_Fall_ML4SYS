from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
import subprocess
import re
from extract_for_blocks import extract_for_blocks
from utils import replace_numbers

# Step 1: Initialize the tokenizer with a WordPiece model
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Configure normalization and pre-tokenization
tokenizer.normalizer = Sequence([NFD(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

# Step 2: Set up a trainer for the tokenizer with specific vocab size and special tokens
trainer = trainers.WordPieceTrainer(
  vocab_size=15000,  # Set as needed
  special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
)

# Step 3: Load dataset in streaming mode
ds = load_dataset('llvm-ml/ComPile', split='train', streaming=True)

# Step 4: Batch processing for tokenizer training
batch_size = 1000  # Process 1000 samples at a time
batch = []
count = 0
max_count = 20000
replace_number = True

for i, module in enumerate(ds):
  bitcode_module = module['content']  # 이미 bytes 형식임

  # 4. llvm-dis 명령어로 비트코드를 텍스트 LLVM-IR로 변환
  dis_command = ['llvm-dis', '-']
  with subprocess.Popen(
      dis_command,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT
  ) as process:
    ir_output, _ = process.communicate(input=bitcode_module)  # encoding 불필요
  decoded_ir = ir_output.decode('utf-8')
  text = "\n\n".join(extract_for_blocks(decoded_ir, True))
  if replace_number is True:
    text = replace_numbers(text)
  batch.append(text)
  count += 1
  
  # Train in batches
  if len(batch) >= batch_size:
    tokenizer.train_from_iterator(batch, trainer=trainer)
    batch = []  # Clear batch after processing
  
  if count >= max_count:
    break
  

# Process any remaining data in the last batch
if batch:
  tokenizer.train_from_iterator(batch, trainer=trainer)

# Save the trained tokenizer
tokenizer.save("./llvm_ir_wordpiece_tokenizer_replaced.json")
