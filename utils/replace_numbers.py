import re

# Replace number with [NUM] for better representation
def replace_numbers(text):
  # return re.sub(r'\b\d+\b', '[NUM]', text)
  return re.sub(r'(?<=\D)\d+|\b\d+\b', '[NUM]', text)

