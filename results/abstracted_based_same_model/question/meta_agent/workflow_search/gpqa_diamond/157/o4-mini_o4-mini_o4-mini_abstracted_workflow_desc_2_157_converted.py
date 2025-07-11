async def forward_157(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Identify and classify the structural and functional domains of the transcription factor (transactivation domain vs. dimerization domain) and pinpoint the locations of mutations X and Y in those domains.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, identifying domains, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    N = self.max_sc
    cot_sc_instruction = 'Sub-task 2: Analyze the normal activation mechanism of the transcription factor: its inactive state, phosphorylation on Ser residues in the transactivation domain, subsequent dimerization, nuclear shuttling, and role in gene transcription.'
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, analyzing activation mechanism, thinking: {thinking2.content}; answer: {answer2.content}')
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
    cot_reflect_instruction = 'Sub-task 3: Characterize the effects of mutation X (recessive loss-of-function in the transactivation domain) and mutation Y (heterozygous dominant-negative in the dimerization domain), integrating domain locations and mutation types.'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_reflect_instruction, 'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1', 'thinking of subtask 2', 'answer of subtask 2'], 'agent_collaboration': 'Reflexion'}
    thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent3.id}, characterizing mutation effects, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 'please review the characterization and provide its limitations.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent3(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent3.id}, refining characterization, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: Define the molecular consequences typically associated with dominant-negative mutations in dimerization domains, such as formation of nonfunctional heterodimers or protein aggregation.'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3'], 'agent_collaboration': 'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, defining molecular consequences, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    debate_instruction5 = 'Sub-task 5: Map each provided answer choice to the molecular consequences of a dominant-negative dimerization-domain mutation, identifying which phenotypes align or conflict.'
    debate_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction5, 'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4'], 'agent_collaboration': 'Debate'}
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction5, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, mapping choices, thinking: {thinking5.content}; answer: {answer5.content}')
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on mapping phenotypes.', is_sub_task=True)
    agents.append(f'Final Decision agent, making mapping decision, thinking: {thinking5.content}; answer: {answer5.content}')
    sub_tasks.append(f'Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}')
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_instruction6 = 'Sub-task 6: Select the single most likely molecular phenotype from the choices based on the mapping in subtask 5 and prepare the final answer.'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_instruction6, 'context': ['user query', 'thinking of subtask 5', 'answer of subtask 5'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent6.id}, selecting final phenotype, thinking: {thinking6.content}; answer: {answer6.content}')
    sub_tasks.append(f'Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}')
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs