async def forward_14(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction = 'Sub-task 1: Extract and list all known system parameters from the query: orbital periods P1 and P2, host star masses M1* and M2*, host star radii R1* and R2*, and note that orbits are circular and planet masses are equal.'
    cot_agent1 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent1.id}, extracting system parameters, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Subtask 1 answer: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Identify the relevant physical relationships for transit probability and orbital semi-major axis: geometric transit probability P_transit ≃ R_star/a for circular orbits, and Kepler’s third law a ∝ (M_star)^(1/3) P^(2/3).'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Self-Consistency CoT Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, identifying relationships, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinking_mapping[answer2_i.content] = thinking2_i
        answer_mapping[answer2_i.content] = answer2_i
    most_common = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[most_common]
    answer2 = answer_mapping[most_common]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Subtask 2 answer: ', sub_tasks[-1])
    cot_instruction3 = 'Sub-task 3: Calculate the semi-major axes a1 and a2 using Kepler\'s third law a ∝ (M_star)^(1/3) P^(2/3), using extracted P1, P2, M1*, and M2*.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, calculating semi-major axes, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Subtask 3 answer: ', sub_tasks[-1])
    cot_reflect_instruction4 = 'Sub-task 4: Compute geometric transit probabilities P_transit1 = R_star1/a1 and P_transit2 = R_star2/a2 using the radii and semi-major axes obtained.'
    cot_agent4 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent4.id}, computing transit probabilities, thinking: {thinking4.content}; answer: {answer4.content}')
    for i in range(N_max):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], 'Critically evaluate the computed transit probabilities for accuracy.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction4, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent4.id}, refining transit probabilities, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Subtask 4 answer: ', sub_tasks[-1])
    debate_instruction5 = 'Sub-task 5: Calculate the ratio P_transit1/P_transit2 to determine which planet has the higher probability to transit.'
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                input_infos5 = [taskInfo, thinking4, answer4]
            else:
                input_infos5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
            thinking5_i, answer5_i = await agent(input_infos5, debate_instruction5, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, ratio calculation, thinking: {thinking5_i.content}; answer: {answer5_i.content}')
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make a final decision on which planet has the higher transit probability based on the ratio.', is_sub_task=True)
    agents.append(f'Final Decision agent, deciding ratio outcome, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    print('Subtask 5 answer: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer