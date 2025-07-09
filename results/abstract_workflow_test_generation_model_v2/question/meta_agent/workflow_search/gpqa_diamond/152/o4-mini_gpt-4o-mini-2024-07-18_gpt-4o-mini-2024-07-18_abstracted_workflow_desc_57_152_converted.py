async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parse the query text to identify the three separate Michael addition reactions, listing their reactants, reagents, and conditions for A, B, C."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing reactions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_sc_instruction = "Sub-task 2: For each reaction A, B, and C, draw or represent the structures of the starting materials (nucleophile and α,β-unsaturated electrophile)."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "output of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, representing structures, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2.content)
        thinkingmapping2[answer2.content] = thinking2
        answermapping2[answer2.content] = answer2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[answer2_content]
    answer2 = answermapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    cot_reflect_instruction = "Sub-task 3: Identify the key mechanistic event for a Michael addition to establish product features."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "outputs of subtask_1", "outputs of subtask_2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, identifying mechanistic event, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Please review the identified mechanistic features and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback round {i}, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, feedback3], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining mechanistic event, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    cot_instruction4 = "Sub-task 4: Parse each answer choice to extract proposed products for A, B, and C, naming functional groups and skeletons."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, parsing answer choices, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    cot_sc_instruction5 = "Sub-task 5: For each proposed product, check connectivity and tautomeric plausibility."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_answers5 = []
    thinkingmapping5 = {}
    answermapping5 = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "context": ["outputs of subtask_4"], "agent_collaboration": "SC_CoT"}
    for i in range(N5):
        thinking5, answer5 = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, checking products, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers5.append(answer5.content)
        thinkingmapping5[answer5.content] = thinking5
        answermapping5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinkingmapping5[answer5_content]
    answer5 = answermapping5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    cot_sc_instruction6 = "Sub-task 6: Assess the impact of reaction conditions on tautomer preference and side transformations."
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["outputs of subtask_5"], "agent_collaboration": "SC_CoT"}
    for i in range(N6):
        thinking6, answer6 = await cot_agents6[i]([taskInfo, thinking5, answer5], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, assessing conditions impact, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers6.append(answer6.content)
        thinkingmapping6[answer6.content] = thinking6
        answermapping6[answer6.content] = answer6
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    debate_instruction7 = "Sub-task 7: Select the answer choice whose products best match mechanistic and thermodynamic preferences."
    debate_agents7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N7 = self.max_round
    all_thinking7 = [[] for _ in range(N7)]
    all_answer7 = [[] for _ in range(N7)]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": debate_instruction7, "context": ["outputs of subtask_6"], "agent_collaboration": "Debate"}
    for r in range(N7):
        for i, agent in enumerate(debate_agents7):
            if r == 0:
                thinking7_agent, answer7_agent = await agent([taskInfo, thinking6, answer6], debate_instruction7, r, is_sub_task=True)
            else:
                thinking7_agent, answer7_agent = await agent([taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1], debate_instruction7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting choice, thinking: {thinking7_agent.content}; answer: {answer7_agent.content}")
            all_thinking7[r].append(thinking7_agent)
            all_answer7[r].append(answer7_agent)
    final_decision_agent7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent7([taskInfo] + all_thinking7[-1] + all_answer7[-1], "Sub-task 7: Make final decision on the correct answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent7.id}, finalizing choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs