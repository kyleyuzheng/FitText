from typing import Dict, Any

DESCRIPTION_MAX = 256  # keep single source of truth

def build_finish_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "Finish",
            "description": (
                "If you believe that you have obtained a result that can answer the task, "
                "please call this function to provide the final answer. Alternatively, if you "
                "recognize that you are unable to proceed with the task in the current state, "
                "call this function to restart. Remember: you must ALWAYS call this function at "
                "the end of your attempt, and the only part that will be shown to the user is the "
                "final answer, so it should contain sufficient information to fully resolve the task "
                "without requiring follow-up questions or clarification, the user will not be able to follow up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {"type": "string", "enum": ["give_answer","give_up_and_restart"]},
                    "final_answer": {
                        "type": "string",
                        "description": 'The final answer if "return_type" == "give_answer".'
                    }
                },
                "required": ["return_type"]
            },
        }
    }

TYPE_MAP = {
    "NUMBER": "integer",
    "STRING": "string",
    "BOOLEAN": "boolean",
}

def build_function_template() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }