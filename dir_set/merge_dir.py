import os
import shutil

def merge_directories(source_dirs, target_dir):
    """
    동일한 디렉토리 구조를 가진 두 디렉토리를 병합합니다.

    :param source_dirs: 병합할 소스 디렉토리 목록 (리스트)
    :param target_dir: 병합된 데이터를 저장할 타겟 디렉토리
    """
    # 타겟 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            # 상대 경로 계산
            relative_path = os.path.relpath(root, source_dir)
            target_path = os.path.join(target_dir, relative_path)

            # 하위 디렉토리가 없으면 생성
            if not os.path.exists(target_path):
                os.makedirs(target_path)

            # 파일 복사
            for file in files:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_path, file)

                # 파일 이름 중복 처리
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(target_file):
                        target_file = os.path.join(target_path, f"{base}_{count}{ext}")
                        count += 1

                # 파일 복사
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")

# 병합 실행
source_dirs = ["A", "B"]  # 병합할 디렉토리 리스트
target_dir = "C"       # 병합된 데이터를 저장할 디렉토리
merge_directories(source_dirs, target_dir)
