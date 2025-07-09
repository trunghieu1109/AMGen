async def forward_158(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Parse the quasar’s spectral data to identify the observed peak wavelength (790 nm) and note the significant flux drop at wavelengths shorter than 790 nm.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, parsing spectral data, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 2: Extract and list the cosmological parameters H0 = 70 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7 for a flat universe, and the four comoving distance options: 6 Gpc, 7 Gpc, 8 Gpc, 9 Gpc. Do NOT select any distance yet.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'CoT'}
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting parameters, thinking: {thinking2.content}; answer: {answer2.content}')
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction = 'Sub-task 3: Determine the quasar’s redshift by associating the observed break at 790 nm with the Lyman-alpha rest wavelength (121.6 nm) and computing z = (790 / 121.6) - 1.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, computing redshift, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    sc_instruction = 'Sub-task 4: Compute the comoving distance DC for the derived redshift using the flat Lambda-CDM formula DC = (c/H0) ∫0→z [dz_prime / sqrt(Omega_m*(1+z_prime)**3 + Omega_Lambda)], with c = 3×10^5 km/s and H0 = 70 km/s/Mpc. Perform a numerical integration, compare the result to the provided options (6, 7, 8, 9 Gpc), select the closest match, and perform a brief sanity check against known DC(z≈5.5) ≈ 7.8 Gpc.'
    N = self.max_sc
    sc_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': sc_instruction, 'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking4_i, answer4_i = await sc_agents[i]([taskInfo, thinking2, answer2, thinking3, answer3], sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {sc_agents[i].id}, computing comoving distance, thinking: {thinking4_i.content}; answer: {answer4_i.content}')
        possible_answers.append(answer4_i.content)
        thinking_mapping[answer4_i.content] = thinking4_i
        answer_mapping[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinking_mapping[answer4_content]
    answer4 = answer_mapping[answer4_content]
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs