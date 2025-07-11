async def forward_150(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Compute the Euclidean norm of the state vector (-1, 2, 1)'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        'subtask_id': 'subtask_1',
        'instruction': cot_instruction,
        'context': ['user query'],
        'agent_collaboration': 'CoT'
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, computed norm, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {
        'thinking': thinking1,
        'answer': answer1
    }
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Normalize the state vector (-1, 2, 1) using the norm from Sub-task 1'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        'subtask_id': 'subtask_2',
        'instruction': cot_sc_instruction,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1'],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, normalized state, thinking: {thinking2_i.content}; answer: {answer2_i.content}')
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f'Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}')
    subtask_desc2['response'] = {
        'thinking': thinking2,
        'answer': answer2
    }
    logs.append(subtask_desc2)
    print('Step 2: ', sub_tasks[-1])

    cot_instruction3 = 'Sub-task 3: Determine the eigenvector of P for eigenvalue 0 by solving (P - 0 I) phi = 0'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        'subtask_id': 'subtask_3',
        'instruction': cot_instruction3,
        'context': ['user query'],
        'agent_collaboration': 'CoT'
    }
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent3.id}, determined eigenvector, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {
        'thinking': thinking3,
        'answer': answer3
    }
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])

    cot_reflect_instruction = 'Sub-task 4: Calculate the probability of measuring eigenvalue 0 by computing |<phi|psi>|^2'
    cot_agent4 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        'subtask_id': 'subtask_4',
        'instruction': cot_reflect_instruction,
        'context': ['user query', 'thinking of subtask 2', 'answer of subtask 2', 'thinking of subtask 3', 'answer of subtask 3'],
        'agent_collaboration': 'Reflexion'
    }
    thinking4, answer4 = await cot_agent4(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent4.id}, initial probability calculation, thinking: {thinking4.content}; answer: {answer4.content}')
    for i in range(N_max):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4], 'Please review the probability calculation and identify any errors.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent4.id}, feedback round {i}, thinking: {feedback4.content}; correct: {correct4.content}')
        if correct4.content == 'True':
            break
        cot_inputs.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent4.id}, refined probability calculation, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {
        'thinking': thinking4,
        'answer': answer4
    }
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs