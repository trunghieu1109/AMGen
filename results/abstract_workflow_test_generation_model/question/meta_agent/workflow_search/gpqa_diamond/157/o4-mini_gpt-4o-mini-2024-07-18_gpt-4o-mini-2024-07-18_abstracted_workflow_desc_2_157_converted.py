async def forward_157(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = 'Sub-task 1: Parse the question and extract key elements: inactive transcription factor, phosphorylation activation, transactivation vs. dimerization domains, mutation X (recessive LOF) and mutation Y (heterozygous dominant-negative).'
    cot_agent1 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction1, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, parsing question, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction2 = 'Sub-task 2: Define the molecular consequence of mutation X in the transactivation domain (missense, recessive loss-of-function only in homozygotes).'
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible2 = []
    think_map2 = {}
    answer_map2 = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction2, 'context': ['user query', 'Sub-task 1 reasoning', 'Sub-task 1 answer'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, defining mutation X, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible2.append(answer2_i.content)
        think_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible2).most_common(1)[0][0]
    thinking2 = think_map2[answer2_content]
    answer2 = answer_map2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])
    cot_instruction3 = 'Sub-task 3: Define the molecular consequence of mutation Y in the dimerization domain (heterozygous dominant-negative effect interfering with wild-type partner).'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_instruction3, 'context': ['user query', 'Sub-task 2 reasoning', 'Sub-task 2 answer'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, defining mutation Y, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    cot_instruction4 = 'Sub-task 4: List all four answer choices and categorize each by the phenotype described (dimerization loss, aggregation, degradation, conformational gain-of-function).'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': cot_instruction4, 'context': ['user query', 'answer choices'], 'agent_collaboration': 'CoT'}
    thinking4, answer4 = await cot_agent4([taskInfo], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, categorizing choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    cot_reflect_instruction5 = 'Sub-task 5: Analyze the dominant-negative mechanism of mutation Y: how a mutant and wild-type subunit form heterodimers and impair function.'
    cot_agent5 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_reflect_instruction5, 'context': ['user query', 'Sub-task 3 reasoning', 'Sub-task 3 answer'], 'agent_collaboration': 'Reflexion'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3], cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial analysis, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], 'Review the analysis and identify limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == 'True':
            break
        thinking5, answer5 = await cot_agent5([taskInfo, thinking3, answer3, thinking5, answer5, feedback5], cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refined analysis, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print('Step 5: ', sub_tasks[-1])
    cot_instruction6 = 'Sub-task 6: List possible cellular fates of nonfunctional mutant–wild-type heterodimers: aggregation/sequestration vs. proteasomal degradation.'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot_instruction6, 'context': ['user query', 'Sub-task 5 reasoning', 'Sub-task 5 answer'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, listing fates, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print('Step 6: ', sub_tasks[-1])
    cot_instruction7 = 'Sub-task 7: Predict which fate (aggregation or degradation) is canonical for dominant-negative dimerization mutants, citing standard examples like p53 tetramer mutants.'
    cot_agent7 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc7 = {'subtask_id': 'subtask_7', 'instruction': cot_instruction7, 'context': ['user query', 'Sub-task 6 reasoning', 'Sub-task 6 answer'], 'agent_collaboration': 'CoT'}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, predicting canonical fate, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {'thinking': thinking7, 'answer': answer7}
    logs.append(subtask_desc7)
    print('Step 7: ', sub_tasks[-1])
    cot_instruction8 = 'Sub-task 8: Map the predicted fate to the categorized answer choices, providing explicit justification for selecting aggregation over degradation based on canonical mechanism.'
    cot_agent8 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc8 = {'subtask_id': 'subtask_8', 'instruction': cot_instruction8, 'context': ['user query', 'Sub-task 4 reasoning', 'Sub-task 4 answer', 'Sub-task 7 reasoning', 'Sub-task 7 answer'], 'agent_collaboration': 'CoT'}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking4, answer4, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, mapping fate to choices, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {'thinking': thinking8, 'answer': answer8}
    logs.append(subtask_desc8)
    print('Step 8: ', sub_tasks[-1])
    cot_reflect_instruction9 = 'Sub-task 9: Critically evaluate the mapping: Does dominant-negative dimerization typically degrade the wild-type or sequester it into inactive complexes? Provide evidence.'
    cot_agent9 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent9 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    subtask_desc9 = {'subtask_id': 'subtask_9', 'instruction': cot_reflect_instruction9, 'context': ['user query', 'Sub-task 8 reasoning', 'Sub-task 8 answer'], 'agent_collaboration': 'Reflexion'}
    thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8], cot_reflect_instruction9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent9.id}, initial critique, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(self.max_round):
        feedback9, correct9 = await critic_agent9([taskInfo, thinking9, answer9], 'Review the mapping evaluation and identify limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent9.id}, feedback: {feedback9.content}; correct: {correct9.content}")
        if correct9.content == 'True':
            break
        thinking9, answer9 = await cot_agent9([taskInfo, thinking8, answer8, thinking9, answer9, feedback9], cot_reflect_instruction9, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent9.id}, refined critique, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {'thinking': thinking9, 'answer': answer9}
    logs.append(subtask_desc9)
    print('Step 9: ', sub_tasks[-1])
    cot_sc_instruction10 = 'Sub-task 10: Generate multiple independent reasoning paths to confirm the mapping decision is robust regarding dominant-negative aggregation-mediated loss-of-function.'
    N2 = self.max_sc
    cot_agents10 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible10 = []
    think_map10 = {}
    answer_map10 = {}
    subtask_desc10 = {'subtask_id': 'subtask_10', 'instruction': cot_sc_instruction10, 'context': ['user query', 'Sub-task 9 reasoning', 'Sub-task 9 answer'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N2):
        thinking10_i, answer10_i = await cot_agents10[i]([taskInfo, thinking9, answer9], cot_sc_instruction10, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents10[i].id}, confirming mapping, thinking: {thinking10_i.content}; answer: {answer10_i.content}")
        possible10.append(answer10_i.content)
        think_map10[answer10_i.content] = thinking10_i
        answer_map10[answer10_i.content] = answer10_i
    answer10_content = Counter(possible10).most_common(1)[0][0]
    thinking10 = think_map10[answer10_content]
    answer10 = answer_map10[answer10_content]
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {'thinking': thinking10, 'answer': answer10}
    logs.append(subtask_desc10)
    print('Step 10: ', sub_tasks[-1])
    cot_instruction11 = 'Sub-task 11: Select and return the letter (A, B, C, or D) corresponding to the answer choice that best fits dominant-negative aggregation-mediated loss-of-function.'
    cot_agent11 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc11 = {'subtask_id': 'subtask_11', 'instruction': cot_instruction11, 'context': ['user query', 'Sub-task 10 reasoning', 'Sub-task 10 answer'], 'agent_collaboration': 'CoT'}
    thinking11, answer11 = await cot_agent11([taskInfo, thinking10, answer10], cot_instruction11, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent11.id}, selecting final letter, thinking: {thinking11.content}; answer: {answer11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking11.content}; answer - {answer11.content}")
    subtask_desc11['response'] = {'thinking': thinking11, 'answer': answer11}
    logs.append(subtask_desc11)
    print('Step 11: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking11, answer11, sub_tasks, agents)
    return final_answer, logs