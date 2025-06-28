async def forward(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Sub-task 1: Identify the relationship between elements of set A and sets B.
    cot_instruction = "Sub-task 1: Identify the relationship between elements of set A and sets B, where the maximum element of B is in A."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying relationship, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')

    # Sub-task 2: Calculate the number of sets B for each element in A.
    cot_sc_instruction = "Sub-task 2: Calculate the number of sets B for each element in A, given the maximum element of B is in A."
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}

    for i in range(N):
        thinking2, answer2 = cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, calculating number of sets B, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2

    answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2]
    answer2 = answermapping[answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')

    # Sub-task 3: Determine the total number of sets B for the entire set A.
    cot_reflect_instruction = "Sub-task 3: Determine the total number of sets B for the entire set A and ensure it matches 2024."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, determining total number of sets B, thinking: {thinking3.content}; answer: {answer3.content}')

    for i in range(N_max):
        feedback, correct = critic_agent([taskInfo, thinking3, answer3], "please review the total number of sets B calculation and provide feedback about its limitations.", i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining total number of sets B, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')

    # Sub-task 4: Determine possible values of elements in set A.
    debate_instruction_4 = "Sub-task 4: Determine possible values of elements in set A that satisfy the condition from sub-task 3."
    debate_agents_4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_3)]
    all_answer4 = [[] for _ in range(N_max_3)]

    for r in range(N_max_3):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1])
                
            thinking4, answer4 = agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, determining possible values of A, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_answers = [ans.content for ans in all_answer4[-1]]
    final_answercontent = Counter(final_answers).most_common(1)[0][0]
    index = final_answers.index(final_answercontent)
    thinking4 = all_thinking4[-1][index]
    answer4 = all_answer4[-1][index]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')

    # Sub-task 5: Calculate the sum of the elements of set A.
    final_decision_agent_5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = final_decision_agent_5([taskInfo, thinking4, answer4], "Sub-task 5: Calculate the sum of the elements of set A.", is_sub_task=True)
    agents.append(f'Final Decision agent, calculating sum of elements of A, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer