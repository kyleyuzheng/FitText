FORMAT_TOOL_FUNCTIONARITY_FUNCTION = """
You are provided with information about an API, including the API's name, description, document, as well as the description of the tool this API belongs to.
Based solely on this information, concisely generate a clear, brief description of the API's functionality. Avoid including any unnecessary information unrelated to the API itself.
"""

FORMAT_TOOL_FUNCTIONARITY_USER_FUNCTION = """
API Name: {api_name}.
API Description: {api_description}.
Name of the Tool this api belongs to: {tool_name}.
Description of the Tool this api belongs to: {tool_description}.
API documentation:: {api_doc}.
Please remember, tell me the functionality of this api without outputting unnecessary information,
ONLY describing the API functionality in short description.
"""