import json
import tiktoken

import re

# token price: https://openai.com/api/pricing/
#We use the fomula: price=price per input token*input token number+price per output token*output token number

import re
import tiktoken

# Constants for pricing
price_per_input_token = 2.5 / 1_000_000  # $2.5 per million input tokens
price_per_output_token = 10 / 1_000_000  # $10 per million output tokens

price_per_input_token2 = 0.15/ 1_000_000  # $0.15 per million input tokens
price_per_output_token2 = 0.6 / 1_000_000  # $0.6 per million output tokens

input_file = "log/gpt-4o-mini/gsm8k_auto_3.log"
output_file = "gpt-4o-mini-output"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

#delete the content from pred_after to GT
pattern = r"pred_after : .*?GT : .*?\n?"
cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL)


with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_content)

encoding_output = tiktoken.encoding_for_model("gpt-4o-mini")  
tokens_output = encoding_output.encode(cleaned_content)       
token_count_output = len(tokens_output)                      

print(f"saved to {output_file}")
print(f"number of tokens in output:{token_count_output}")


file_path = "demo/gsm8k_3"  

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

encoding_demo = tiktoken.encoding_for_model("gpt-4")  
tokens_demo = encoding_demo.encode(content)               
token_count_demo = len(tokens_demo)  

print(f"number of tokens in demo(input){token_count_demo}")

# Calculate the total cost
total_cost_4o = (
    token_count_demo * 1319 * price_per_input_token + token_count_output * price_per_output_token
)

total_cost_4o_mini = (
    token_count_demo * 1319 * price_per_input_token2 + token_count_output * price_per_output_token2
)
#print(f"cost for gpt-4o: {total_cost_4o}") #$24.601575
print(f"cost for gpt-4o-mini: {total_cost_4o_mini}")  #$1.7855359499999999
