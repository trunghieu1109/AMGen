async def forward_188(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Extract candidate particles from the query and define the evaluation criterion as association with spontaneously-broken symmetry.'
    cot_agent = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1,answer1 = await cot_agent([taskInfo],cot_instruction,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, extracting candidates, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    print('Step 1: ', sub_tasks[-1])
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    cot_sc_instruction = 'Sub-task 2: For each particle (magnon, skyrmion, pion, phonon), determine whether it is associated with a spontaneously-broken symmetry in its physical context, and record a brief justification.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_sc_instruction,'context':['user query','thinking of subtask 1','answer of subtask 1'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking2,answer2 = await cot_agents[i]([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, evaluating association cases, thinking: {thinking2.content}; answer: {answer2.content}')
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    print('Step 2: ', sub_tasks[-1])
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    cot_instruction3 = 'Sub-task 3: Identify the particle not associated with any spontaneously-broken symmetry based on the recorded justifications.'
    cot_agent3 = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_instruction3,'context':['user query','thinking of subtask 2','answer of subtask 2'],'agent_collaboration':'CoT'}
    thinking3,answer3 = await cot_agent3([taskInfo,thinking2,answer2],cot_instruction3,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, identifying non-associated particle, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    print('Step 3: ', sub_tasks[-1])
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    cot_instruction4 = 'Sub-task 4: Map the identified particle to its corresponding answer letter (A, B, C, or D).'
    cot_agent4 = LLMAgentBase(['thinking','answer'],'Chain-of-Thought Agent',model=self.node_model,temperature=0.0)
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_instruction4,'context':['user query','thinking of subtask 3','answer of subtask 3'],'agent_collaboration':'CoT'}
    thinking4,answer4 = await cot_agent4([taskInfo,thinking3,answer3],cot_instruction4,is_sub_task=True)
    agents.append(f'CoT agent {cot_agent4.id}, mapping to letter, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    print('Step 4: ', sub_tasks[-1])
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4,answer4,sub_tasks,agents)
    return final_answer, logs