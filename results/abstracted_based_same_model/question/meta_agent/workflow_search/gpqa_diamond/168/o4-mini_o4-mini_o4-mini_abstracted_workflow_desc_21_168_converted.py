async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parse the decay query to extract key features: the original decay 2A -> 2B + 2E + 2V, continuous energy spectrum of E with endpoint Q, and variant decay description emitting M instead of 2V."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Identify the kinematic principle governing the shape and endpoint of a beta-type energy spectrum, noting dependence on the number of final-state particles and their masses."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing kinematic principle, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Enumerate and compare final-state particle counts and masses: original has 2E + 2V, variant has 2E + 1M, noting that M is massless."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, comparing final states, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction = "Sub-task 4: Analyze how replacing two massive particles V with one massless particle M changes the degrees of freedom and kinematic constraints on the E spectrum."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking4 = [[] for _ in range(N_max)]
    all_answer4 = [[] for _ in range(N_max)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing degrees of freedom, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on kinematic constraints impact.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing analysis of degrees of freedom, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Determine whether the variant decay spectrum of E remains continuous or becomes discrete based on the degrees of freedom after substitution."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, determining spectrum continuity, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_sc_instruction6 = "Sub-task 6: Determine the direction of endpoint energy change (increase or decrease) by applying energy conservation and kinetic energy arguments for the variant decay."
    M = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(M)]
    possible_answers6 = []
    thinking_mapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_sc_instruction6, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "SC_CoT"}
    for i in range(M):
        thinking6, answer6 = await cot_agents6[i]([taskInfo, thinking4, answer4], cot_sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, analyzing endpoint shift, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers6.append(answer6.content)
        thinking_mapping6[answer6.content] = thinking6
        answermapping6[answer6.content] = answer6
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinking_mapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_instruction7 = "Sub-task 7: Map the conclusions about spectrum continuity/discreteness and endpoint shift to one of the four provided multiple-choice answers (A, B, C, or D)."
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction7, "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, mapping to multiple-choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs