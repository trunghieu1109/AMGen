async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = 'Sub-task 1: Extract electron speed v and position uncertainty Δx from the query.'
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        'subtask_id': 'subtask_1',
        'instruction': cot_instruction,
        'context': ['user query'],
        'agent_collaboration': 'CoT'
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting inputs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    v = 2 * 10**8
    dx = 0.1 * 10**-9
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = 'Sub-task 2: Identify the value of the reduced Planck constant ħ for later use.'
    N = self.max_sc
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {
        'subtask_id': 'subtask_2',
        'instruction': cot_sc_instruction,
        'context': ['user query'],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying ħ, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmapping[answer2_i.content] = thinking2_i
        answermapping[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    hbar = 1.054571817e-34
    print("Step 2: ", sub_tasks[-1])

    cot_instruction3 = 'Sub-task 3: State the Heisenberg uncertainty principle and derive the expression for Δp = ħ/(2·Δx).'
    cot_agent3 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        'subtask_id': 'subtask_3',
        'instruction': cot_instruction3,
        'context': ['user query', thinking1.content, answer1.content, thinking2.content, answer2.content],
        'agent_collaboration': 'CoT'
    }
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, deriving Δp expression, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction4 = 'Sub-task 4: Compute the numerical value of Δp = ħ/(2·Δx) using ħ and Δx.'
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_answers4 = []
    thinkingmapping4 = {}
    answermapping4 = {}
    subtask_desc4 = {
        'subtask_id': 'subtask_4',
        'instruction': cot_sc_instruction4,
        'context': ['user query', thinking3.content, answer3.content],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, computing Δp, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinkingmapping4[answer4_i.content] = thinking4_i
        answermapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinkingmapping4[answer4_content]
    answer4 = answermapping4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    dp = hbar/(2*dx)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction5 = 'Sub-task 5: Calculate the minimum energy uncertainty ΔE ≈ v·Δp using v and Δp.'
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {
        'subtask_id': 'subtask_5',
        'instruction': cot_sc_instruction5,
        'context': ['user query', thinking4.content, answer4.content],
        'agent_collaboration': 'SC_CoT'
    }
    for i in range(N5):
        thinking5_i, answer5_i = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, computing ΔE, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers5.append(answer5_i.content)
        thinkingmapping5[answer5_i.content] = thinking5_i
        answermapping5[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    dE = v * dp
    print("Step 5: ", sub_tasks[-1])

    cot_reflect_instruction6 = 'Sub-task 6: Compare the computed energy uncertainty to the provided choices and select the closest match (A–D).'
    cot_agent6 = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        'subtask_id': 'subtask_6',
        'instruction': cot_reflect_instruction6,
        'context': ['user query', thinking5.content, answer5.content],
        'agent_collaboration': 'Reflexion'
    }
    thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, selecting choice, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], 'please review the choice selection and provide its limitations.', i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == 'True':
            break
        cot_inputs6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent6(cot_inputs6, cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining choice selection, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs