async def forward_20(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Characterize any two-digit integer n in base b by introducing digit variables x and y and expressing n = x·b + y with digit bounds 1 ≤ x ≤ b–1 and 0 ≤ y ≤ b–1.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f'CoT agent {cot_agent.id}, characterizing two-digit integer, thinking: {thinking1.content}; answer: {answer1.content}')
    sub_tasks.append(f'Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}')
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print('Step 1: ', sub_tasks[-1])
    cot_sc_instruction = 'Sub-task 2: Derive and restate the b-beautiful condition in terms of x, y, and s: since n = s^2 and √n = s must equal x + y, obtain the key equation (x + y)^2 = x·b + y with s = x + y, 1 ≤ x ≤ b–1, 0 ≤ y ≤ b–1.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query','thinking of subtask 1','answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, i, is_sub_task=True)
        agents.append(f'CoT-SC agent {cot_agents[i].id}, deriving b-beautiful condition, thinking: {thinking2.content}; answer: {answer2.content}')
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
    cot_reflect_instruction = 'Sub-task 3: For a fixed base b ≥ 2, enumerate all b-beautiful integers by listing integer sums s from ceil(sqrt(b)) to b-1; for each s and each x in [1, b-1], set y = s−x and check 0 ≤ y ≤ b-1 and (x + y)^2 = x·b + y; collect n = s^2 for all valid (x,y,s). Then validate the enumeration against the b-beautiful definition.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot_reflect_instruction, 'context': ['user query','thinking of subtask 1','answer of subtask 1','thinking of subtask 2','answer of subtask 2'], 'agent_collaboration': 'Reflexion'}
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f'Reflexion CoT agent {cot_agent.id}, enumerating and validating b-beautiful integers, thinking: {thinking3.content}; answer: {answer3.content}')
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], 'please review the enumeration and validation of b-beautiful integers for this base and provide feedback.', i, is_sub_task=True)
        agents.append(f'Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}')
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f'Reflexion CoT agent {cot_agent.id}, refining enumeration, thinking: {thinking3.content}; answer: {answer3.content}')
    sub_tasks.append(f'Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}')
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print('Step 3: ', sub_tasks[-1])
    debate_instruction = 'Sub-task 4: Search for the smallest integer base b ≥ 2 whose enumeration from Sub-task 3 yields more than ten b-beautiful integers by iterating b starting from 2, invoking Sub-task 3 logic to count valid n for each b, and stopping when the count exceeds 10. Return that base b.'
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': debate_instruction, 'context': ['user query','thinking of subtask 3','answer of subtask 3'], 'agent_collaboration': 'Debate'}
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f'Debate agent {agent.id}, round {r}, proposing base, thinking: {thinking4.content}; answer: {answer4.content}')
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], 'Sub-task 4: Decide the smallest base b from proposed candidates.', is_sub_task=True)
    agents.append(f'Final Decision agent, deciding base, thinking: {thinking4.content}; answer: {answer4.content}')
    sub_tasks.append(f'Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}')
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print('Step 4: ', sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs