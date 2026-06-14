DIVERSITY_PROMPT='''This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous.'''

END_PROMPT='''You reach the maximum number of steps or queries, you should stop retrieving or calling new functions.
Instead, summarize what you have learned so far, try to answer the user's question as best as possible using the available information, and then call:
Finish -> give_answer to give your final answer.'''


