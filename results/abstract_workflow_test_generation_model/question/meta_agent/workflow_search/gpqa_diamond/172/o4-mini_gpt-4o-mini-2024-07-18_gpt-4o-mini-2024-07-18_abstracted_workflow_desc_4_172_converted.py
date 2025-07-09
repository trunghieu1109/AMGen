async def forward_172(self, taskInfo):
    from collections import Counter
    print('Task Requirement: ', taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = 'Sub-task 1: Identify and record the physical constants electron mass m = 9.11×10⁻³¹ kg and reduced Planck constant ħ = 1.055×10⁻³⁴ J·s'
    cot_agent = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id':'subtask_1','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying constants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print('Step 1: ', sub_tasks[-1])
    subtask_desc1['response'] = {'thinking':thinking1,'answer':answer1}
    logs.append(subtask_desc1)
    cot_instruction = 'Sub-task 2: Extract and record the given experimental parameters: position uncertainty Δx = 0.1 nm (convert to 1.0×10⁻¹⁰ m) and electron speed v = 2×10⁸ m/s'
    cot_agent2 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc2 = {'subtask_id':'subtask_2','instruction':cot_instruction,'context':['user query'],'agent_collaboration':'CoT'}
    thinking2, answer2 = await cot_agent2([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, extracting parameters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print('Step 2: ', sub_tasks[-1])
    subtask_desc2['response'] = {'thinking':thinking2,'answer':answer2}
    logs.append(subtask_desc2)
    cot_sc_instruction = 'Sub-task 3: Compute the minimum momentum uncertainty Δp = ħ/(2·Δx). Show the formula, substitute numeric values with units, and produce Δp in kg·m/s'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc3 = {'subtask_id':'subtask_3','instruction':cot_sc_instruction,'context':['user query','answer of subtask_1','answer of subtask_2'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking3_i, answer3_i = await cot_agents[i]([taskInfo, answer1, answer2], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing Δp, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_answers.append(answer3_i.content)
        thinkingmapping[answer3_i.content] = thinking3_i
        answermapping[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers).most_common(1)[0][0]
    thinking3 = thinkingmapping[answer3_content]
    answer3 = answermapping[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print('Step 3: ', sub_tasks[-1])
    subtask_desc3['response'] = {'thinking':thinking3,'answer':answer3}
    logs.append(subtask_desc3)
    cot_sc_instruction = 'Sub-task 4: Compute the minimum energy uncertainty ΔE = v·Δp. Show the formula, substitute the numeric v and Δp with units, and produce ΔE in joules'
    cot_agents4 = [LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    subtask_desc4 = {'subtask_id':'subtask_4','instruction':cot_sc_instruction,'context':['user query','answer of subtask_2','answer of subtask_3'],'agent_collaboration':'SC_CoT'}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, answer2, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, computing ΔE, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print('Step 4: ', sub_tasks[-1])
    subtask_desc4['response'] = {'thinking':thinking4,'answer':answer4}
    logs.append(subtask_desc4)
    cot_reflect_instruction = 'Sub-task 5: Perform a reflective back-of-the-envelope check on ΔE to verify the order of magnitude, comparing with rough estimate ~5×10⁻²⁵ kg·m/s × 2×10⁸ m/s ≈10⁻¹⁶ J'
    cot_agent5 = LLMAgentBase(['thinking','answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(['feedback','correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking4, answer4]
    subtask_desc5 = {'subtask_id':'subtask_5','instruction':cot_reflect_instruction,'context':['user query','thinking of subtask_4','answer of subtask_4'],'agent_collaboration':'Reflexion'}
    thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, checking ΔE order, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking5, answer5], 'Review the back-of-the-envelope check and verify if ΔE order is consistent', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == 'True':
            break
        cot_inputs.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refining check, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print('Step 5: ', sub_tasks[-1])
    subtask_desc5['response'] = {'thinking':thinking5,'answer':answer5}
    logs.append(subtask_desc5)
    debate_instruction = 'Sub-task 6: Compare the validated ΔE value to the provided choices (~10⁻¹⁹, ~10⁻¹⁸, ~10⁻¹⁶, ~10⁻¹⁷ J) and select the letter (A, B, C, or D) corresponding to the closest match'
    debate_agents = [LLMAgentBase(['thinking','answer'], 'Debate Agent', model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc6 = {'subtask_id':'subtask_6','instruction':debate_instruction,'context':['user query','answer of subtask_5'],'agent_collaboration':'Debate'}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6_i, answer6_i = await agent([taskInfo, answer5], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, answer5] + all_thinking[r-1] + all_answer[r-1]
                thinking6_i, answer6_i = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking[r].append(thinking6_i)
            all_answer[r].append(answer6_i)
    final_decision_agent = LLMAgentBase(['thinking','answer'], 'Final Decision Agent', model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], 'Sub-task 6: Make final decision on the closest match', is_sub_task=True)
    agents.append(f"Final Decision Agent, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print('Step 6: ', sub_tasks[-1])
    subtask_desc6['response'] = {'thinking':thinking6,'answer':answer6}
    logs.append(subtask_desc6)
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs