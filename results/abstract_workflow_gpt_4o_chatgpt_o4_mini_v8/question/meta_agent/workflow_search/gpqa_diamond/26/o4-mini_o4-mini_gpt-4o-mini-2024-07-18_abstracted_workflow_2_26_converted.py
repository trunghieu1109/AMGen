async def forward_26(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction1 = 'Sub-task 1: Identify and classify the ribonucleoprotein particle that binds nascent chains and targets them for processing, based on the query context.'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, classified ribonucleoprotein, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])
    sc_instruction2 = 'Sub-task 2: Based on the classification from Sub-task 1, identify and classify the nascent chain, determining its origin and whether it carries a signal for entry into a specific compartment.'
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents[i]([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents[i].id}, classified nascent chain, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinking_mapping[answer2_i.content] = thinking2_i
        answer_mapping[answer2_i.content] = answer2_i
    majority_answer2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[majority_answer2]
    answer2 = answer_mapping[majority_answer2]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])
    reflect_instruction3 = 'Sub-task 3: Analyze the dialogue clues you really need some sugar and somewhat rough together with the classifications from Sub-tasks 1 and 2 to infer the meeting site of SRP and the nascent chain.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(inputs3, reflect_instruction3, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3.id}, inferring meeting site, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], 'Critically evaluate the inferred meeting site and point out any inconsistencies.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}')
        if correct3.content == 'True':
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(inputs3, reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3.id}, refining meeting site inference, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])
    debate_instruction4 = 'Sub-task 4: Determine the destination of the nascent chain after its interaction with the SRP and select the correct option among the four given choices.'
    debate_agents4 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                input_infos4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(input_infos4, debate_instruction4, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, determining destination, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Make final decision on the chain destination based on the previous debate.', is_sub_task=True)
    agents.append(f'Final Decision agent {final_decision_agent4.id}, selecting final option, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer