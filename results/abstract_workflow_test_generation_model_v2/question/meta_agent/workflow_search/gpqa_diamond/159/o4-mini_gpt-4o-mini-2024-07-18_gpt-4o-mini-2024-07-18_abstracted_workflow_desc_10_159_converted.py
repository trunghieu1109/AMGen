async def forward_159(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Recognize that as N → ∞ the N-sided polygon aperture becomes a circular aperture whose radius equals the common apothem a."
    cot_agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction1, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, recognizing aperture shape, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Derive the diameter D of the equivalent circular aperture in terms of the apothem a (D = 2a)."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, deriving D from apothem, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinkingmapping2[answer2_i.content] = thinking2_i
        answermapping2[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[answer2_content]
    answer2 = answermapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction3 = "Sub-task 3: Compute the angular positions θ₁ and θ₂ of the first two minima for a circular aperture using the first two zeros (α₁, α₂) of the Bessel function J₁: θ_j = α_j·λ/(π D)."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_answers3 = []
    thinkingmapping3 = {}
    answermapping3 = {}
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing θ₁ and θ₂, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_answers3.append(answer3_i.content)
        thinkingmapping3[answer3_i.content] = thinking3_i
        answermapping3[answer3_i.content] = answer3_i
    answer3_content = Counter(possible_answers3).most_common(1)[0][0]
    thinking3 = thinkingmapping3[answer3_content]
    answer3 = answermapping3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction4 = "Sub-task 4: Debate and decide whether the requested angular distance is between ±θ₁ or between θ₂ and θ₁ by comparing both interpretations against the physical meaning of 'first two minima.'"
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(N_max4):
        for agent in debate_agents4:
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(input_infos, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating interpretation, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on interpretation.", is_sub_task=True)
    agents.append(f"Final Decision agent, choosing interpretation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Based on the correct interpretation (difference between successive minima), calculate the angular separation Δθ = |θ₂ – θ₁|."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4, thinking3, answer3], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, calculating Δθ, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Substitute D = 2a into Δθ = (α₂ – α₁)·λ/(π D) and simplify to a numeric coefficient times λ/a."
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["user query", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents6:
        thinking6_i, answer6_i = await agent([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, simplifying Δθ formula, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
        possible_answers6.append(answer6_i.content)
        thinkingmapping6[answer6_i.content] = thinking6_i
        answermapping6[answer6_i.content] = answer6_i
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_sc_instruction7 = "Sub-task 7: Compare the simplified result to the provided answer choices (1.220 λ/a, 0.610 λ/a, 0.500 λ/a, 0.506 λ/a) and select the matching letter."
    N7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible_answers7 = []
    thinkingmapping7 = {}
    answermapping7 = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction7, "context": ["user query", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents7:
        thinking7_i, answer7_i = await agent([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, selecting matching option, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible_answers7.append(answer7_i.content)
        thinkingmapping7[answer7_i.content] = thinking7_i
        answermapping7[answer7_i.content] = answer7_i
    answer7_content = Counter(possible_answers7).most_common(1)[0][0]
    thinking7 = thinkingmapping7[answer7_content]
    answer7 = answermapping7[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs