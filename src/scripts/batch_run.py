import os,argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True)
args = parser.parse_args()
# epochs = [1,5,10,30]
epochs=[50]
# dups = [1,2,3,4,5]
dups=[1]
exps = ["vanilla","diagonal_double","concat","sum"]
'''
python run_all.py --exp vanilla & ; python run_all.py --exp diagonal_double & ; python run_all.py --exp concat & ; python run_all.py --exp sum & ; wait
'''

exp_command_0_dict= {
    "vanilla":"echo vanilla",
    "diagonal_double":"echo diagonal_double",
    "concat":"set -xg DIAGONAL_DOUBLE_VARIANT_CONCAT concat",
    "sum":"set -xg DIAGONAL_DOUBLE_VARIANT_SUM sum",
}

exp_command_1_dict= {
    "vanilla":"diagonal",
    "diagonal_double":"diagonal_double",
    "concat":"diagonal_double",
    "sum":"diagonal_double",
}
exp_id = exps.index(args.exp)
commands = []
for epoch in epochs:
    for dup in dups:
        exp_str = f"2_by_2_mult_double_10k_py_{args.exp}_exp{dup}_epochs{epoch}"
        data_str = f"2_by_2_mult_double_10k"
        command = " \n " .join([
            f"cd /workspaces/implicit_chain_of_thought",
            f"conda activate icot",
            f"set -xg CUDA_VISIBLE_DEVICES {exp_id}",
            f"{exp_command_0_dict[args.exp]}",
            f"mkdir -p generation_logs/{exp_str}/gpt2",
            f"mkdir -p train_models/{exp_str}/gpt2",
            f"python src/train_teacher.py --train_path data/{data_str}/train.txt --val_path data/{data_str}/valid.txt --epochs {epoch} --lr 5e-5 --batch_size 32 --base_model gpt2 --save_model train_models/{exp_str}/gpt2/teacher 2>&1 >generation_logs/{exp_str}/gpt2/teacher.txt",
            f"python src/train_mind_reading_student.py --train_path data/{data_str}/train.txt --val_path data/{data_str}/valid.txt --epochs {epoch} --lr 5e-5 --batch_size 32 --base_model gpt2 --teacher train_models/{exp_str}/gpt2/teacher/checkpoint_{epoch-1} --save_model train_models/{exp_str}/gpt2/student_initial --delta dynamic 2>&1 >generation_logs/{exp_str}/gpt2/student.txt",
            f"python src/train_thought_emulator.py --train_path data/{data_str}/train.txt --val_path data/{data_str}/valid.txt --epochs {epoch} --lr 5e-5 --batch_size 32 --base_model gpt2 --teacher train_models/{exp_str}/gpt2/teacher/checkpoint_{epoch-1} --save_model train_models/{exp_str}/gpt2/emulator_initial --delta dynamic --subset {exp_command_1_dict[args.exp]} --mixture_size 1 2>&1 >generation_logs/{exp_str}/gpt2/emulator.txt",
            f"python src/train_coupled_emulator_and_student.py --train_path data/{data_str}/train.txt --val_path data/{data_str}/valid.txt --epochs {epoch} --lr 5e-5 --batch_size 32 --student train_models/{exp_str}/gpt2/student_initial/checkpoint_{epoch-1} --emulator train_models/{exp_str}/gpt2/emulator_initial/checkpoint_{epoch-1} --save_model train_models/{exp_str}/gpt2/ 2>&1 >generation_logs/{exp_str}/gpt2/coupled.txt",
            f"mkdir -p saved_models/{exp_str}/gpt2/student",
            f"mkdir -p saved_models/{exp_str}/gpt2/emulator",
            f"cp -r train_models/{exp_str}/gpt2/student/checkpoint_{epoch-1} saved_models/{exp_str}/gpt2/student/checkpoint_{epoch-1}",
            f"cp -r train_models/{exp_str}/gpt2/emulator/checkpoint_{epoch-1} saved_models/{exp_str}/gpt2/emulator/checkpoint_{epoch-1}",
            f"command rm -rf train_models/{exp_str}/gpt2/student_initial train_models/{exp_str}/gpt2"
        ])
        # print(f" fish -c '{command}'")
        os.system(f" fish -c '{command}'")