import csv
import subprocess
from datasets import load_dataset
import ir2vec

# 1. ComPile 데이터셋을 스트리밍 방식으로 불러오기
ds = load_dataset('llvm-ml/ComPile', split='train', streaming=True)

# 2. CSV 파일로 저장할 파일을 열기
with open("embedding_vectors_ir2vec.csv", "w", newline='') as csvfile:
    # CSV writer 객체 생성
    csvwriter = csv.writer(csvfile)
    
    # CSV의 헤더 작성
    csvwriter.writerow(["ir_id", "embedding_vector"])
    
    # 3. 데이터셋을 순회하면서 첫 100개의 모듈만 처리
    for i, module in enumerate(ds):
        if i >= 10:
            break  # 20,000개까지만 처리
        
        # if i < 19454:
        #     continue
        
        if i in [985, 1454, 2038, 3136, 3240, 3426, 4503, 6588, 6723, 7167, 8137, 8482, 9070, 9283, 10559
                , 10868, 12108, 12169, 12718, 13274, 13593, 13620, 13642, 13909, 13987, 14210, 14735
                , 15923, 17014, 17624, 18466, 18775, 19454]:
            continue
        try:
            bitcode_module = module['content']  # 이미 bytes 형식임
            dis_command = ['llvm-dis', '-']
            with subprocess.Popen(
                dis_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            ) as process:
                ir_output, _ = process.communicate(input=bitcode_module)

            with open("ir_temp.ll", "w") as temp_file:
                temp_file.write(ir_output.decode('utf-8'))
            initObj = ir2vec.initEmbedding("ir_temp.ll", "fa", "p")
            program_vector = initObj.getProgramVector()
            csvwriter.writerow([i, program_vector])

            # 진행 상황 확인용 출력
            print(f"Processed module {i}")

        except Exception as e:
            print(f"Error processing module {i}: {e}")
            continue
