async def forward_24(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Rewrite each given logarithmic equation in terms of a = log_2 x, b = log_2 y, c = log_2 z to convert the system into linear equations involving a, b, and c." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, rewrite logarithmic equations as linear system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Formulate the system of linear equations explicitly from Sub-task 1, representing each logarithmic equation as a linear equation in variables a, b, and c." 
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, formulate linear equations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction_3a = "Sub-task 3a: Add all three linear equations from Sub-task 2 to derive the sum equation involving a + b + c." 
    cot_sc_instruction_3b = "Sub-task 3b: Add the first and second equations from Sub-task 2 to form an equation involving a and b, then solve for c." 
    cot_sc_instruction_3c = "Sub-task 3c: Add the first and third equations from Sub-task 2 to form an equation involving a and c, then solve for b." 
    cot_sc_instruction_3d = "Sub-task 3d: Add the second and third equations from Sub-task 2 to form an equation involving b and c, then solve for a." 
    cot_sc_instruction_3e = "Sub-task 3e: Verify that the values of a, b, and c obtained satisfy the sum equation from Sub-task 3a and all original linear equations to ensure consistency and positivity constraints." 
    N = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3e = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, add all equations to get sum equation, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    possible_answers_3b = []
    thinkingmapping_3b = {}
    answermapping_3b = {}
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, add first and second equations and solve for c, thinking: {thinking3b.content}; answer: {answer3b.content}")
        possible_answers_3b.append(answer3b.content)
        thinkingmapping_3b[answer3b.content] = thinking3b
        answermapping_3b[answer3b.content] = answer3b
    answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
    thinking3b = thinkingmapping_3b[answer3b_content]
    answer3b = answermapping_3b[answer3b_content]
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    possible_answers_3c = []
    thinkingmapping_3c = {}
    answermapping_3c = {}
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_sc_instruction_3c,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3c, answer3c = await cot_agents_3c[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, add first and third equations and solve for b, thinking: {thinking3c.content}; answer: {answer3c.content}")
        possible_answers_3c.append(answer3c.content)
        thinkingmapping_3c[answer3c.content] = thinking3c
        answermapping_3c[answer3c.content] = answer3c
    answer3c_content = Counter(possible_answers_3c).most_common(1)[0][0]
    thinking3c = thinkingmapping_3c[answer3c_content]
    answer3c = answermapping_3c[answer3c_content]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {
        "thinking": thinking3c,
        "answer": answer3c
    }
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    possible_answers_3d = []
    thinkingmapping_3d = {}
    answermapping_3d = {}
    subtask_desc3d = {
        "subtask_id": "subtask_3d",
        "instruction": cot_sc_instruction_3d,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3d, answer3d = await cot_agents_3d[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3d[i].id}, add second and third equations and solve for a, thinking: {thinking3d.content}; answer: {answer3d.content}")
        possible_answers_3d.append(answer3d.content)
        thinkingmapping_3d[answer3d.content] = thinking3d
        answermapping_3d[answer3d.content] = answer3d
    answer3d_content = Counter(possible_answers_3d).most_common(1)[0][0]
    thinking3d = thinkingmapping_3d[answer3d_content]
    answer3d = answermapping_3d[answer3d_content]
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
    subtask_desc3d['response'] = {
        "thinking": thinking3d,
        "answer": answer3d
    }
    logs.append(subtask_desc3d)
    print("Step 3d: ", sub_tasks[-1])
    possible_answers_3e = []
    thinkingmapping_3e = {}
    answermapping_3e = {}
    subtask_desc3e = {
        "subtask_id": "subtask_3e",
        "instruction": cot_sc_instruction_3e,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a", "thinking of subtask 3b", "answer of subtask 3b", "thinking of subtask 3c", "answer of subtask 3c", "thinking of subtask 3d", "answer of subtask 3d"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking3e, answer3e = await cot_agents_3e[i]([taskInfo, thinking3a, answer3a, thinking3b, answer3b, thinking3c, answer3c, thinking3d, answer3d], cot_sc_instruction_3e, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3e[i].id}, verify solutions for a,b,c satisfy all equations and positivity, thinking: {thinking3e.content}; answer: {answer3e.content}")
        possible_answers_3e.append(answer3e.content)
        thinkingmapping_3e[answer3e.content] = thinking3e
        answermapping_3e[answer3e.content] = answer3e
    answer3e_content = Counter(possible_answers_3e).most_common(1)[0][0]
    thinking3e = thinkingmapping_3e[answer3e_content]
    answer3e = answermapping_3e[answer3e_content]
    sub_tasks.append(f"Sub-task 3e output: thinking - {thinking3e.content}; answer - {answer3e.content}")
    subtask_desc3e['response'] = {
        "thinking": thinking3e,
        "answer": answer3e
    }
    logs.append(subtask_desc3e)
    print("Step 3e: ", sub_tasks[-1])
    cot_instruction_4 = "Sub-task 4: Calculate log_2(x^4 y^3 z^2) using the verified values of a, b, and c from Sub-task 3e by applying logarithmic properties." 
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3e", "answer of subtask 3e"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3e, answer3e], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, compute log_2(x^4 y^3 z^2), thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Compute the absolute value |log_2(x^4 y^3 z^2)| and express it as a reduced fraction m/n where m and n are relatively prime positive integers." 
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute absolute value and reduce fraction, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the reduced fraction and compute m+n.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final reduced fraction and sum m+n, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction_6 = "Sub-task 6: Find the sum m + n from the fraction obtained in Sub-task 5, which is the final integer answer to be returned." 
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, compute sum m+n from fraction, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs