import os
import dotenv

# check if api key is set in environment variables
print(dotenv.load_dotenv())

print('OPENAI_API_KEY' in os.environ)
print(repr(os.getenv('OPENAI_API_KEY')))
