async def forward_197(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Identify all relevant cobalt(II)-thiocyanato species and their stability constants β1 through β4 based on the solution description.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying species and constants, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Write the formation equilibrium expressions for each complex using βₙ = [Co(SCN)ₙ^(2–n)] / ([Co²⁺][SCN⁻]^n) for n = 1…4.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, writing equilibrium expressions, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction3 = 'Sub-task 3: Formulate the total-cobalt mass-balance equation and express each complex concentration in terms of [Co2+], [SCN-], and βₙ.'
    cot_agent3 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, formulating mass-balance, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_sc_instruction4 = 'Sub-task 4: Derive the expression for the fraction α₂ = β₂[SCN-]² / (1 + β₁[SCN-] + β₂[SCN-]² + β₃[SCN-]³ + β₄[SCN-]⁴).'
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_sc_instruction4, 'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N4):
        thinking4, answer4 = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents4[i].id}, deriving alpha2 expression, thinking: {thinking4.content}; answer: {answer4.content}')
        possible_answers4.append(answer4.content)
        thinkingmapping4[answer4.content] = thinking4
        answermapping4[answer4.content] = answer4
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    cot_sc_instruction5 = 'Sub-task 5: Substitute c(Co)=1e-2 M, [SCN-]=0.1 M and β values into α₂ expression and compute its numerical value.'
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_sc_instruction5, 'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N5):
        thinking5, answer5 = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents5[i].id}, computing numerical alpha2, thinking: {thinking5.content}; answer: {answer5.content}')
        possible_answers5.append(answer5.content)
        thinkingmapping5[answer5.content] = thinking5
        answermapping5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_instruction6 = 'Sub-task 6: Convert the computed fraction α₂ into a percentage by multiplying by 100.'
    cot_agent6 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_instruction6, 'context': ['user query', 'thinking of subtask 5', 'answer of subtask 5'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, converting to percentage, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    debate_instruction7 = 'Sub-task 7: Compare the calculated percentage to the provided choices and select the closest match letter.'
    debate_agents7 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N7 = self.max_round
    all_thinking7 = [[] for _ in range(N7)]
    all_answer7 = [[] for _ in range(N7)]
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': debate_instruction7, 'context': ['user query', 'thinking of subtask 6', 'answer of subtask 6'], 'agent_collaboration': 'Debate'}
    for r in range(N7):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                thinking7, answer7 = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                inputs7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(inputs7, debate_instruction7, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, selecting closest match, thinking: {thinking7.content}; answer: {answer7.content}')
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent7 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], 'Sub-task 7: Make final decision on the closest choice.', is_sub_task=True)
    agents.append(f'Final Decision agent, making final selection, thinking: {thinking7.content}; answer: {answer7.content}')
    sub_tasks.append(f'Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}')
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs