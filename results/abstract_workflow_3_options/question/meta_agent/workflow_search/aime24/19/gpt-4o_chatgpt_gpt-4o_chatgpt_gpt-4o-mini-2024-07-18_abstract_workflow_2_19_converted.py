async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Stage 1: Condition Identification and Modular Reasoning
    # Sub-task 1: Identify and compute the properties of the 13th root of unity
    cot_instruction = "Sub-task 1: Identify and compute the properties of the 13th root of unity, focusing on the fact that the sum of all 13th roots of unity is zero."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying properties of 13th root of unity, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Simplify the expression 2 - 2ω^k + ω^2k for each k from 0 to 12
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, simplify the expression 2 - 2ω^k + ω^2k for each k from 0 to 12 using the properties of roots of unity."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}

    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, simplifying expression, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Stage 2: Intermediate Inference and Validation
    # Sub-task 3: Compute the product of the simplified expressions
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, compute the product of the simplified expressions for k = 0 to 12."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, computing product, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the product calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining product calculation, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Apply modular arithmetic to find the remainder when divided by 1000
    debate_instruction_4 = "Sub-task 4: Based on the output of sub-task 3, apply modular arithmetic to find the remainder of the product when divided by 1000."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_3)]
    all_answer4 = [[] for _ in range(N_max_3)]

    for r in range(N_max_3):
        for agent in debate_agents_4:
            thinking4, answer4 = agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, applying modular arithmetic, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_answers = [ans.content for ans in all_answer4[-1]]
    final_answercontent = Counter(final_answers).most_common(1)[0][0]
    index = final_answers.index(final_answercontent)
    thinking4 = all_thinking4[-1][index]
    answer4 = all_answer4[-1][index]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer