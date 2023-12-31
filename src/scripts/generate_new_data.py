import os
import argparse
import random
import logging


# complete: 1 3 4 5 * 8 1 9 3||8 4 4 3 4 + 0 1 3 4 5 0 ( 8 5 7 7 9 0 ) + 0 0 9 7 8 8 4 ( 8 5 6 5 8 9 4 ) + 0 0 0 3 9 2 6 1 #### 8 5 6 8 7 2 1 2
# readable: 5431*3918||43448+54310(97758)+4887900(4985858)+16293000####21278658


def target_format_value(value, value_length, reverse_digits):
    '''
    value: the integer value to be formatted
    value_length: the length of the formatted value
    reverse_digits: whether to reverse the digits of the value

    example: target_format_value(123,5,True) -> "3 2 1 0 0"
    '''
    value_digits_str = [d for d in str(value)]
    if reverse_digits:
        value_digits_str.reverse()
        value_digits_str.extend(["0"] * (value_length - len(value_digits_str)))
    else:
        value_digits_str.reverse()
        value_digits_str.extend(["0"] * (value_length - len(value_digits_str)))
        value_digits_str.reverse()
    return " ".join(value_digits_str)

def generate_input(int_a,int_b,reverse_digits):
    '''
    int_a: the first integer
    int_b: the second integer
    reverse_digits: whether to reverse the digits of the value

    example: generate_input(123,456,True) -> "3 2 1 0 0 * 6 5 4 0 0"
    '''
    length_a = len(str(int_a))
    length_b = len(str(int_b))
    assert length_a == length_b
    result = f"{target_format_value(int_a,length_a,reverse_digits)} * {target_format_value(int_b,length_b,reverse_digits)}"
    return result

def generate_readable_chain_of_thought(int_a, int_b, reverse_digits):
    digits_a = [int(d) for d in str(int_a)]
    len_digits_a = len(digits_a)
    digits_b = [int(d) for d in str(int_b)]
    len_digits_b = len(digits_b)
    assert len_digits_a == len_digits_b

    readable_chain_of_thought = ""
    target_chain_of_thought = ""

    current_sum = 0
    for i, digit_b in reversed([*enumerate(digits_b)]):
        digit_b_value = int(digit_b) * 10 ** (len_digits_b - 1 - i)
        add_value = digit_b_value * int_a
        current_sum += add_value
        if i == len_digits_b - 1:
            readable_chain_of_thought += f"{str(add_value)}"
            target_chain_of_thought += target_format_value(
                current_sum, len_digits_a + (len_digits_b - i), reverse_digits
            )
        elif i == 0:
            readable_chain_of_thought += f" + {str(add_value)}"
            target_chain_of_thought += (
                f" + {str(target_format_value(add_value,len_digits_a+(len_digits_b-i),reverse_digits))}"
            )
        else:
            readable_chain_of_thought += f" + {str(add_value)}({current_sum})"
            target_chain_of_thought += f" + {str(target_format_value(add_value,len_digits_a+(len_digits_b-i),reverse_digits))} ( {target_format_value(current_sum,len_digits_a+(len_digits_b-i),reverse_digits)} )"
    return target_chain_of_thought

def generate_result(int_a, int_b, reverse_digits):
    int_result = int_a * int_b
    digits_a = [int(d) for d in str(int_a)]
    len_digits_a = len(digits_a)
    digits_b = [int(d) for d in str(int_b)]
    len_digits_b = len(digits_b)
    string_result = target_format_value(int_result, len_digits_a+len_digits_b , reverse_digits)
    return string_result

def generate_line(int_a_list, int_b_list, reverse_digits, expression_number=1):
    multiplication_expression_list = []
    chain_of_thought_list = []
    result_list = []
    if isinstance(int_a_list,int):
        int_a_list = [int_a_list]
    if isinstance(int_b_list,int):
        int_b_list = [int_b_list]
    assert len(int_a_list) == len(int_b_list)
    
    for i in range(expression_number):
        multiplication_expression = generate_input(int_a_list[i], int_b_list[i], reverse_digits)
        multiplication_expression_list.append(multiplication_expression)
        chain_of_thought = generate_readable_chain_of_thought(int_a_list[i], int_b_list[i], reverse_digits)
        chain_of_thought_list.append(chain_of_thought)
        result = generate_result(int_a_list[i], int_b_list[i], reverse_digits)
        result_list.append(result)
    final_multiplication_expression = ", ".join(multiplication_expression_list)
    final_chain_of_thought = ", ".join(chain_of_thought_list)
    final_result = ", ".join(result_list)
    line = f"{final_multiplication_expression}||{final_chain_of_thought} #### {final_result}"
    return line


def generate_data(
    digit_number_A: int,
    digit_number_B: int,
    expression_number: int,
    dataset_size: int,
    reverse_digits: bool,
    excluded_file_path_list: str = None,
):
    excluded_lines = []
    for excluded_file_path in excluded_file_path_list:
        print(f"reading {excluded_file_path}")
        with open(excluded_file_path, "r") as f:
            excluded_lines.extend(f.read().split("\n"))
            if excluded_lines[-1] == "":
                excluded_lines.pop()
    logging.debug("excluded_lines: {}".format(excluded_lines))

    lines = []
    random_seed = random.randint(0, 999999999)
    logging.info("random seed: {}".format(random_seed))
    random.seed(random_seed)
    for i in range(dataset_size):
        accepted: bool = False
        while not accepted:
            int_a_list = [random.randint(10**(digit_number_A-1), 10**(digit_number_A)-1) for _ in range(expression_number)]
            int_b_list = [random.randint(10**(digit_number_B-1), 10**(digit_number_B)-1) for _ in range(expression_number)]
            line = generate_line(int_a_list, int_b_list, reverse_digits,expression_number)
            if line not in excluded_lines:
                accepted = True
            else:
                pass
                logging.warning("rejected: {}".format(line))
            lines.append(line)
    return lines

def build_parser():
    parser = argparse.ArgumentParser(description="Generate new data for training")
    parser.add_argument(
        "-da", "--digit_number_a", type=int, help="number of digits", default=4
    )
    parser.add_argument(
        "-db", "--digit_number_b", type=int, help="number of digits", default=4
    )
    parser.add_argument(
        "-en", "--expression_number", type=int, help="number of expressions", default=1
    )
    parser.add_argument(
        "-ds", "--dataset_size", type=int, help="size of dataset", default=4
    )
    parser.add_argument(
        "-rd", "--reverse_digits", type=bool, help="reverse digits", default=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output file", default=None
    )
    parser.add_argument(
        "-p","--print",action="store_true",help="print to stdout",default=False
    )
    parser.add_argument(
        "-e","--excluded",help="excluded file path list",default=[], type = str,nargs="*"
    )
    return parser


def main(args=None):
    args = args or build_parser().parse_args()
    if isinstance(args.excluded,str):
        args.excluded = [args.excluded]
    print(args)
    if not args.output and not args.print:
        raise ValueError("Either --output or --print must be set")
    generated_data_list = generate_data(
        args.digit_number_a,
        args.digit_number_b,
        args.expression_number,
        args.dataset_size,
        args.reverse_digits,
        args.excluded
    )
    if args.print:
        for line in generated_data_list:
            print(line)
    if args.output:
        with open(args.output, "w") as f:
            for line in generated_data_list:
                f.write(line + "\n")




if __name__ == "__main__":
    
    logging.basicConfig(level=getattr(logging,os.getenv("LOG_LEVEL", "INFO").upper()))
    # print(f"log level is {loggi}")
    assert generate_line(5431,3918,1) == "1 3 4 5 * 8 1 9 3||8 4 4 3 4 + 0 1 3 4 5 0 ( 8 5 7 7 9 0 ) + 0 0 9 7 8 8 4 ( 8 5 6 5 8 9 4 ) + 0 0 0 3 9 2 6 1 #### 8 5 6 8 7 2 1 2"
    main()
