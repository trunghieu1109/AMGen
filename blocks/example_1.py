import inspect

async def forward(self, taskInfo):
    from collections import Counter
    # Initialize lists to keep track of sub-tasks and agents
    sub_tasks = []
    agents = []

    # Sub-task 1: Calculate Aya's walking speed, s
    cot_instruction = "Sub-task 1: Calculate Aya's walking speed s based on the information that when she walks at s km/h, the walk takes 4 hours including t minutes spent in the coffee shop."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, calculating speed s, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Determine the time spent in the coffee shop, t
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, determine the time spent in the coffee shop t using the information that when she walks at s+2 km/h, the walk takes 2 hours and 24 minutes including t minutes in the coffee shop."
    N = global_max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, determining time t, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2]
    answer2 = answer_mapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Calculate the time it takes for Aya to walk 9 km at s+1/2 km/h
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, calculate the time it takes for Aya to walk 9 km at s+1/2 km/h, including the t minutes spent in the coffee shop."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=global_node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=global_node_model, temperature=0.0)
    N_max = global_max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, calculating final time, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], cot_reflect_instruction, i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining final time, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer

func_string = inspect.getsource(forward)

EXAMPLE_1 = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "Example_1",
    "code": """{func_string}""".format(func_string=func_string)

}
