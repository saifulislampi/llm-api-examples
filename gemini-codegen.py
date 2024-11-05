import json
import google.generativeai as genai
import time
import os
import argparse


def gemini_response(prompt, temperature, token_limit, model_name):
    try:
        model = genai.GenerativeModel(
            model_name, system_instruction="You are a helpful coding assistant. Only output the generated code."
        )

        generation_config = genai.GenerationConfig(
            max_output_tokens=token_limit,
            temperature=temperature,
            candidate_count=1
        )

        response = model.generate_content(
            prompt['prompt'],
            generation_config=generation_config
        )

        prompt['output'] = response.to_dict()
        time.sleep(7)  # Add delay to respect rate limits
        return prompt

    except Exception as e:
        print(f"An error occurred: {e}")
        prompt['output'] = str(e)
        time.sleep(30)  # Wait before retrying in case of API rate limits
        return prompt


def main():
    parser = argparse.ArgumentParser(description="Process a single prompt with the Gemini API.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file containing the API key.')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to the file containing the single prompt in JSON format.')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help='List of temperatures to use.')
    parser.add_argument('--token_limits', type=int, nargs='+', default=[512], help='List of token limits to use.')
    parser.add_argument('--gemini_model_name', type=str, default='gemini-1.5-flash-002', help='Name of the Gemini model to use.')
    args = parser.parse_args()

    # Load the API key and Configure the API
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            API_KEY = config['GEMINI_API_KEY']
        genai.configure(api_key=API_KEY)
    except (FileNotFoundError, KeyError):
        print(f"Error: API key configuration issue.")
        exit(1)

    # Load the single prompt
    try:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prompt file '{args.prompt_file}' not found.")
        exit(1)

    gemini_model_name = args.gemini_model_name

    # Process the prompt for each temperature and token limit
    for temp in args.temperatures:
        for token_limit in args.token_limits:
            print(f'Processing prompt with temperature {temp} and token limit {token_limit}')
            
            # Generate response
            updated_prompt = gemini_response(prompt, temp, token_limit, gemini_model_name)

            # Prepare output directory and filename
            output_folder = "./Output/gemini_responses"
            os.makedirs(output_folder, exist_ok=True)
            output_filename = f"{output_folder}/{gemini_model_name}_Output_T{temp}_Tokens{token_limit}.json"

            # Save the updated prompt with response to a file
            with open(output_filename, "w") as f_out:
                json.dump(updated_prompt, f_out, indent=4)
                print(f'Saved to {output_filename}')


if __name__ == '__main__':
    main()

