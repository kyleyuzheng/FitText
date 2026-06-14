import re

BEGIN = "<|begin_func_description|>"
END = "<|end_func_description|>"

FUNC_DESC_PATTERN = re.compile(rf'{re.escape(BEGIN)}(.*?){re.escape(END)}', re.DOTALL) # r f string, DOTALL for multi-line capture