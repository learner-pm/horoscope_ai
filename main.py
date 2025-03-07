import subprocess
import sys

def run_script(script_name, description):
    """通过 subprocess 执行脚本并打印描述"""
    print(f"开始执行 {description} ...")
    try:
        subprocess.check_call([sys.executable, script_name])
        print(f"{description} 执行完成！\n")
    except subprocess.CalledProcessError as e:
        print(f"执行 {description} 失败：{e}")
        sys.exit(1)

def main():
    # 执行 data_preprocess.py
    run_script("src/data_preprocess.py", "分词处理")

    # 执行 train.py
    run_script("src/train.py", "训练模型")

    # 执行 generate.py
    run_script("src/generate.py","使用模型生成文本")

if __name__ == "__main__":
    main()
