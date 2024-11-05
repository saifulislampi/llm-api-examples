import json
import openai
import time
import os
import argparse


def gpt_response(prompt, temperature, token_limit):
    try:
        prompt_content = prompt['prompt']

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. Only output the generated code."
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            temperature=temperature,
            max_tokens=token_limit,
            n=1
        )
        prompt['output'] = response['choices'][0]['message']['content']
        time.sleep(1)
        return prompt

    except Exception as e:
        print(f"An error occurred: {e}")
        prompt['output'] = str(e)
        time.sleep(30)
        return prompt


def main():
    parser = argparse.ArgumentParser(description="Process a single prompt with the GPT API.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file containing the API key.')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the file containing the single prompt in JSON format.')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help='List of temperatures to use.')
    parser.add_argument('--token_limits', type=int, nargs='+', default=[512], help='List of token limits to use.')
    args = parser.parse_args()

    # Load API key and configure OpenAI
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            openai.api_key = config['OPENAI_KEY']
    except (FileNotFoundError, KeyError):
        print("Error: API key configuration issue.")
        exit(1)

    # Load the single prompt
    try:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompt file '{args.prompt_file}' not found.")
        exit(1)

    # Process the prompt for each temperature and token limit
    for temp in args.temperatures:
        for token_limit in args.token_limits:
            print(f'Processing prompt with temperature {temp} and token limit {token_limit}')
            
            # Generate response
            updated_prompt = gpt_response(prompt, temp, token_limit)

            # Prepare output directory and filename
            output_folder = "./Output/gpt_responses"
            os.makedirs(output_folder, exist_ok=True)
            output_filename = f"{output_folder}/GPT3.5_Output_T{temp}_Tokens{token_limit}.json"

            # Save the updated prompt with response to a file
            with open(output_filename, "w") as f_out:
                json.dump(updated_prompt, f_out, indent=4)
                print(f'Saved to {output_filename}')


if __name__ == '__main__':
    main()

