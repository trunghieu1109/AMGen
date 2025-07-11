async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Enumerate all unordered line segments of the regular 12-gon, labeling each segment by its endpoint indices (i,j) with 0 ≤ i < j < 12.'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, enumerating segments, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    cot_sc_instruction = 'Sub-task 2: Group each segment from subtask_1 into one of six direction classes d in {1,…,6}, where d ≡ (j−i) mod 12 (with class 6 corresponding to diameters).'
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query','thinking of subtask 1','answer of subtask 1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, grouping segments by direction, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot3_instruction = "Sub-task 3: Identify all unordered perpendicular direction-class pairs (d,d') from subtask_2 satisfying |d−d'| ≡ 3 (mod 6)."
    cot3_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': cot3_instruction, 'context': ['user query','thinking of subtask 2','answer of subtask 2'], 'agent_collaboration': 'CoT'}
    thinking3, answer3 = await cot3_agent([taskInfo, thinking2, answer2], cot3_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot3_agent.id}, identifying perpendicular class pairs, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    N = self.max_sc
    sc_agents4_1 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible4_1 = []
    thinkingmap4_1 = {}
    answermap4_1 = {}
    instruction4_1 = "Sub-task 4.1: For each perpendicular pair from subtask_3, list all ordered pairs of distinct segments within class d and within class d'."
    subtask_desc4_1 = {'subtask_id': 'subtask_4.1', 'instruction': instruction4_1, 'context': ['user query','thinking of subtask 2','answer of subtask 2','thinking of subtask 3','answer of subtask 3'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking4_1_i, answer4_1_i = await sc_agents4_1[i]([taskInfo, thinking2, answer2, thinking3, answer3], instruction4_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents4_1[i].id}, listing segment pairs for perpendicular classes, thinking: {thinking4_1_i.content}; answer: {answer4_1_i.content}")
        possible4_1.append(answer4_1_i.content)
        thinkingmap4_1[answer4_1_i.content] = thinking4_1_i
        answermap4_1[answer4_1_i.content] = answer4_1_i
    answer4_1_content = Counter(possible4_1).most_common(1)[0][0]
    thinking4_1 = thinkingmap4_1[answer4_1_content]
    answer4_1 = answermap4_1[answer4_1_content]
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {'thinking': thinking4_1, 'answer': answer4_1}
    logs.append(subtask_desc4_1)
    print("Step 4: ", sub_tasks[-1])
    instruction4_2 = "Sub-task 4.2: For each combination of one pair from class d and one pair from class d' from subtask_4.1, compute the four intersection points of their supporting lines and determine whether each intersection lies inside the 12-gon."
    cot4_2_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc4_2 = {'subtask_id': 'subtask_4.2', 'instruction': instruction4_2, 'context': ['user query','thinking of subtask 4.1','answer of subtask 4.1'], 'agent_collaboration': 'CoT'}
    thinking4_2, answer4_2 = await cot4_2_agent([taskInfo, thinking4_1, answer4_1], instruction4_2, is_sub_task=True)
    agents.append(f"CoT agent {cot4_2_agent.id}, computing intersections and inside checks, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {'thinking': thinking4_2, 'answer': answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 5: ", sub_tasks[-1])
    instruction4_3 = 'Sub-task 4.3: Filter the combinations from subtask_4.2 to those where the four intersection points are distinct and form right angles, ensuring valid rectangles, and record the count for each perpendicular class pair.'
    cot4_3_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc4_3 = {'subtask_id': 'subtask_4.3', 'instruction': instruction4_3, 'context': ['user query','thinking of subtask 4.2','answer of subtask 4.2'], 'agent_collaboration': 'Reflexion'}
    inputs4_3 = [taskInfo, thinking4_2, answer4_2]
    thinking4_3, answer4_3 = await cot4_3_agent(inputs4_3, instruction4_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot4_3_agent.id}, filtering valid rectangles, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    for i in range(N_max):
        feedback4_3, correct4_3 = await critic_agent([taskInfo, thinking4_3, answer4_3], 'Please review the filtering of valid rectangles and provide its limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on filtering, thinking: {feedback4_3.content}; answer: {correct4_3.content}")
        if correct4_3.content == 'True':
            break
        inputs4_3.extend([thinking4_3, answer4_3, feedback4_3])
        thinking4_3, answer4_3 = await cot4_3_agent(inputs4_3, instruction4_3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot4_3_agent.id}, refining filtering, thinking: {thinking4_3.content}; answer: {answer4_3.content}")
    sub_tasks.append(f"Sub-task 4.3 output: thinking - {thinking4_3.content}; answer - {answer4_3.content}")
    subtask_desc4_3['response'] = {'thinking': thinking4_3, 'answer': answer4_3}
    logs.append(subtask_desc4_3)
    print("Step 6: ", sub_tasks[-1])
    debate_instruction5 = 'Sub-task 5: Sum the rectangle counts recorded in subtask_4.3 over all perpendicular direction pairs to obtain the total number of rectangles.'
    debate_agents5 = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max5)]
    all_answer5 = [[] for _ in range(N_max5)]
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': debate_instruction5, 'context': ['user query','thinking of subtask 4.3','answer of subtask 4.3'], 'agent_collaboration': 'Debate'}
    for r in range(N_max5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4_3, answer4_3], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking4_3, answer4_3] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5_i, answer5_i = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, summing rectangle counts, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_agent5 = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 'Sub-task 5: Make final decision on the total number of rectangles in the dodecagon.', is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent5.id}, determining total rectangles, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print("Step 7: ", sub_tasks[-1])
    cot6_instruction = 'Sub-task 6: Perform a sanity check on the total number of rectangles by comparing against known counts in smaller regular polygons or expected magnitude.'
    cot6_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': cot6_instruction, 'context': ['user query','thinking of subtask 5','answer of subtask 5'], 'agent_collaboration': 'CoT'}
    thinking6, answer6 = await cot6_agent([taskInfo, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot6_agent.id}, sanity checking result, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs