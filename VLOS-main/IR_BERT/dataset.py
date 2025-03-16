from datasets import load_dataset
from torch.utils.data import IterableDataset
import subprocess
from extract_for_blocks import extract_for_blocks
import re

# Replace number with [NUM] for better representation
def replace_numbers(text):
  # return re.sub(r'\b\d+\b', '[NUM]', text)
  return re.sub(r'(?<=\D)\d+|\b\d+\b', '[NUM]', text)



class StreamingIRDataset(IterableDataset):
  def __init__(self, split='train', batch_size=8, num_data=8, replace_number=True, tokenizer=None, start_index=0):
    self.dataset = load_dataset('llvm-ml/ComPile', split=split, streaming=True)
    self.batch_size = batch_size
    self.num_data = num_data
    self.replace_number = replace_number
    self.tokenizer = tokenizer
    self.start_index = start_index

  def preprocess(self, module, i):
    dis_command = ['llvm-dis', '-']
    with subprocess.Popen(
        dis_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ) as process:
      ir_output, _ = process.communicate(input=module['content'])
    for_blocks = extract_for_blocks(ir_output.decode('utf-8'), True)

    preprocess_result = []
    for text in for_blocks:
      if self.replace_number:
        text = replace_numbers(text)
      preprocess_result.append(self.tokenizer(text, padding="max_length", truncation=True, max_length=512))
    for_blocks = []
    return preprocess_result

  def __iter__(self):
    batch = []
    count = 0
    for i, module in enumerate(self.dataset):  # Ensure module is the dataset entry
      if i < self.start_index:
        continue
      tokenized_blocks = self.preprocess(module, i)
      count += 1
      for block in tokenized_blocks:
        batch.append(block)
        if len(batch) >= self.batch_size:
          # print('Yield Batch!!!!!!!!!!!!!!')
          yield batch
          batch = []
      if count >= self.num_data:
          break
      
