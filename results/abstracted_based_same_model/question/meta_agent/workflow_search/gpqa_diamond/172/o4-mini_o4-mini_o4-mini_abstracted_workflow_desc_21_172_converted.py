async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Extract Δx, v, and particle type from the query context."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {'subtask_id': 'subtask_1', 'instruction': cot_instruction, 'context': ['user query'], 'agent_collaboration': 'CoT'}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {'thinking': thinking1, 'answer': answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Identify the Heisenberg uncertainty relation Δx·Δp ≈ ħ/2 based on extracted parameters."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc2 = {'subtask_id': 'subtask_2', 'instruction': cot_sc_instruction, 'context': ['user query', 'thinking1', 'answer1'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying uncertainty principle, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    chosen2 = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[chosen2]
    answer2 = answer_map[chosen2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {'thinking': thinking2, 'answer': answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    sc_instruction3 = "Sub-task 3: Rearrange Δx·Δp ≈ ħ/2 to solve for momentum uncertainty Δp = ħ/(2·Δx)."
    N = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible3 = []
    thinking_map3 = {}
    answer_map3 = {}
    subtask_desc3 = {'subtask_id': 'subtask_3', 'instruction': sc_instruction3, 'context': ['user query', 'thinking2', 'answer2'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking3, answer3 = await sc_agents3[i]([taskInfo, thinking2, answer2], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents3[i].id}, deriving Δp formula, thinking: {thinking3.content}; answer: {answer3.content}")
        possible3.append(answer3.content)
        thinking_map3[answer3.content] = thinking3
        answer_map3[answer3.content] = answer3
    chosen3 = Counter(possible3).most_common(1)[0][0]
    thinking3 = thinking_map3[chosen3]
    answer3 = answer_map3[chosen3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {'thinking': thinking3, 'answer': answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction4 = "Sub-task 4: Compute numerical Δp using ħ = 1.055e-34 J·s and Δx = 0.1 nm and refine the result."
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {'subtask_id': 'subtask_4', 'instruction': debate_instruction4, 'context': ['user query', 'thinking3', 'answer3'], 'agent_collaboration': 'Debate'}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing Δp, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on numerical Δp value.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding Δp, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {'thinking': thinking4, 'answer': answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Use ΔE ≈ v·Δp to express ΔE in terms of computed Δp and given v."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {'subtask_id': 'subtask_5', 'instruction': cot_instruction5, 'context': ['user query', 'thinking4', 'answer4'], 'agent_collaboration': 'CoT'}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, deriving ΔE relation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {'thinking': thinking5, 'answer': answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    sc_instruction6 = "Sub-task 6: Calculate numerical ΔE and select the closest match among choices A–D (~10^-19, ~10^-18, ~10^-16, ~10^-17 J)."
    N = self.max_sc
    sc_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible6 = []
    thinking_map6 = {}
    answer_map6 = {}
    subtask_desc6 = {'subtask_id': 'subtask_6', 'instruction': sc_instruction6, 'context': ['user query', 'thinking5', 'answer5'], 'agent_collaboration': 'SC_CoT'}
    for i in range(N):
        thinking6, answer6 = await sc_agents6[i]([taskInfo, thinking5, answer5], sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents6[i].id}, computing ΔE and selecting choice, thinking: {thinking6.content}; answer: {answer6.content}")
        possible6.append(answer6.content)
        thinking_map6[answer6.content] = thinking6
        answer_map6[answer6.content] = answer6
    chosen6 = Counter(possible6).most_common(1)[0][0]
    thinking6 = thinking_map6[chosen6]
    answer6 = answer_map6[chosen6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {'thinking': thinking6, 'answer': answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs