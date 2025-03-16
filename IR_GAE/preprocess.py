import subprocess
import os

# Preprocess Function
def preprocess(module, ir_index, graph_index):
    # Ensure directories exist
    os.makedirs('ir', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)

    # Generate LLVM IR
    dis_command = ['llvm-dis', '-']
    with subprocess.Popen(
        dis_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ) as process:
        ir_output, _ = process.communicate(input=module['content'])

    ir_file_path = f"ir/ir_{ir_index}.ll"
    with open(ir_file_path, "w") as temp_file:
        temp_file.write(ir_output.decode('utf-8'))

    # Generate call graph
    command = [
        'opt',
        '-passes=dot-callgraph',
        '-disable-output',
        ir_file_path
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # The generated DOT file is named as ir_file_path + '.callgraph.dot'
    generated_dot_file = ir_file_path + '.callgraph.dot'
    new_dot_file_path = f'graphs/graph_{graph_index}.dot'

    if os.path.exists(generated_dot_file):
        os.rename(generated_dot_file, new_dot_file_path)
    else:
        raise FileNotFoundError(f"Expected DOT file {generated_dot_file} not found.")

    os.remove(ir_file_path)

    return new_dot_file_path  # Return the path to the DOT file