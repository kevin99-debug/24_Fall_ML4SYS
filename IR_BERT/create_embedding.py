from transformers import BertModel, BertTokenizerFast
from datasets import load_dataset
import subprocess
from extract_for_blocks import extract_for_blocks
from utils import replace_numbers
import numpy as np
import csv

# Directory where the model was saved
model_directory = "./results/model"  # Replace with your actual directory path

# Load the model and tokenizer
model = BertModel.from_pretrained("./results/model")
tokenizer = BertTokenizerFast(tokenizer_file="llvm_ir_wordpiece_tokenizer_replaced.json",
                              unk_token="[UNK]",
                              pad_token="[PAD]",
                              cls_token="[CLS]",
                              sep_token="[SEP]",
                              mask_token="[MASK]")

if "[NUM]" not in tokenizer.vocab:
  tokenizer.add_tokens(["[NUM]"])

output_csv = "embedding_vectors.csv"
count = 0

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ir_id", "loop_id", "header", "embedding_vector"])

dataset = load_dataset('llvm-ml/ComPile', split='train', streaming=True)
for i, module in enumerate(dataset):
  dis_command = ['llvm-dis', '-']
  with subprocess.Popen(
      dis_command,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT
  ) as process:
    ir_output, _ = process.communicate(input=module['content'])
  for_blocks = extract_for_blocks(ir_output.decode('utf-8'), i)
  # print(for_blocks)

  for j, text in enumerate(for_blocks):
    text = replace_numbers(text)

    header = text.split(":")[0].strip() if ":" in text else ""
    header = header.replace("[NUM]", "").strip()

    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    outputs = model(**tokenized)
    embedding_vector = outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy().flatten()
    
    with open(output_csv, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i, j, header, embedding_vector.tolist()])
    count += 1
    if count % 100 == 0:
       print("Extracting for embedding done: " + str(count))
  if count > 50000:
     break
  for_blocks = []

