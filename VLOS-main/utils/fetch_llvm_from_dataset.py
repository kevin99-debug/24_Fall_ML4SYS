from datasets import load_dataset
import subprocess

ds = load_dataset('llvm-ml/ComPile', split='train', streaming=True)

for i, module in enumerate(ds):
    if i >= 20:
        break

    bitcode_module = module['content']

    dis_command = ['llvm-dis', '-']
    with subprocess.Popen(
        dis_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ) as process:
        ir_output, _ = process.communicate(input=bitcode_module)

    # result_string = extract_for_blocks(ir_output.decode('utf-8'))

    with open("ir/ir_" + str(i) + ".ll", "w") as temp_file:
        temp_file.write(ir_output.decode('utf-8'))

