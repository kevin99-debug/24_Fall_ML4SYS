import re

def extract_for_blocks(ir_code, only_body=False):
    # 블록 시작을 찾기 위한 패턴: 'for'로 시작하는 라벨
    block_pattern = re.compile(r'(\bfor\.\w+\b):')
    for_cond_pattern = re.compile(r'(\bfor\.cond\w*\b):')
    for_end_pattern = re.compile(r'(\bfor\.end\w*\b):')
    for_body_pattern = re.compile(r'(\bfor\.body\w*\b):')
    
    # 블록을 추출할 리스트
    for_blocks = []
    
    lines = ir_code.splitlines()
    capture = False
    current_block = []
    current_for = None
    block_end = False
    nested_loop_count = 0
    ith_for_loop = 1


    for line in lines:
        # 'for' 블록의 시작을 찾음
        if line == "}":
            continue
        elif block_pattern.search(line):
            if only_body == True and for_body_pattern.search(line) == None:
                continue
            for_cond = for_cond_pattern.search(line)
            if for_cond != None and current_for == None:
                current_for = for_cond.group(1)
            for_end = for_end_pattern.search(line)
            if for_end != None and current_for != None:
                if current_for in line:
                    current_for = None
                    block_end = True
                else:
                    nested_loop_count += 1

            if current_block:
                for_blocks.append("\n".join(current_block))
            current_block = [line]  # 새 블록 시작
            capture = True
        elif capture and line.strip() == "":  # 빈 줄이 나오면 블록이 끝났다고 판단
            capture = False
            for_blocks.append("\n".join(current_block))
            if block_end == True:
                block_end = False
                # current_block.append("\n")
                # current_block.append(f"Module: {i} | Nested Loop Count: {nested_loop_count + 1} | ith Loop: {ith_for_loop}")
                nested_loop_count = 0
                ith_for_loop += 1
                # current_block.append("====================================")
                # current_block.append("\n")
                # with open("ir_temp.ll", "w") as temp_file:
                #     temp_file.write("\n\n".join(for_blocks))
                # initObj = ir2vec.initEmbedding("ir_temp.ll", "fa", "p")
                # print(f"Module: {i} | Nested Loop Count: {nested_loop_count + 1} | ith Loop: {ith_for_loop} | vector: {initObj.getProgramVector()}")
                # for_blocks = []
            current_block = []
        elif capture:
            current_block.append(line)

    if current_block:  # 마지막 블록 처리
        for_blocks.append("\n".join(current_block))
    # 추출된 블록을 하나의 문자열로 합침

    return for_blocks
