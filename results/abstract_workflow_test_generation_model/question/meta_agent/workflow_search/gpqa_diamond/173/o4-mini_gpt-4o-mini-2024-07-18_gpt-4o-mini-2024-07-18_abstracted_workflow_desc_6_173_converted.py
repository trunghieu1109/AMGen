async def forward_173(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract numerical parameters: initial rest-mass energy, mass-defect fraction, and rest-mass ratio from the query."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting numerical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction = "Sub-task 2: Compute total kinetic energy release T_release = mass-defect fraction × E0."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing T_release, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction = "Sub-task 3: Determine individual rest masses m1 and m2 given m1:m2=2:1 and sum=0.99 M."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent([taskInfo, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, determining m1 and m2, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction = "Sub-task 4: Compute classical kinetic energy T1_classical = T_release × (m2/(m1+m2))."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction, "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing T1_classical, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    sc_instruction = "Sub-task 5: Solve for the common momentum p using energy conservation T_release = (sqrt(p^2+m1^2)-m1)+(sqrt(p^2+m2^2)-m2)."
    N = self.max_sc
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": sc_instruction, "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_3", "answer of subtask_3"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5_i, answer5_i = await cot_sc_agents[i]([taskInfo, thinking2, answer2, thinking3, answer3], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, solving for momentum p, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
        possible_answers.append(answer5_i.content)
        thinkingmapping[answer5_i.content] = thinking5_i
        answermapping[answer5_i.content] = answer5_i
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5_content]
    answer5 = answermapping[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction = "Sub-task 6: Compute relativistic kinetic energy T1_rel = sqrt(p^2 + m1^2) - m1."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction, "context": ["user query", "thinking of subtask_5", "answer of subtask_5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing T1_rel, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_reflect_instruction = "Sub-task 7: Calculate ΔT = T1_rel - T1_classical and convert result into MeV."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking4, answer4, thinking6, answer6]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "Reflexion"}
    thinking7, answer7 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating ΔT, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max):
        feedback7, correct7 = await critic_agent([taskInfo, thinking7, answer7], "Review calculation of ΔT and mention any limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback on ΔT, thinking: {feedback7.content}; correct: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining ΔT calculation, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    debate_instruction = "Sub-task 8: Compare calculated ΔT (in MeV) with choices [10 MeV, 5 MeV, 2 MeV, 20 MeV] and select the closest match."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_rounds = self.max_round
    all_thinking8 = [[] for _ in range(N_rounds)]
    all_answer8 = [[] for _ in range(N_rounds)]
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": debate_instruction, "context": ["user query", "thinking of subtask_7", "answer of subtask_7"], "agent_collaboration": "Debate"}
    for r in range(N_rounds):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking8_i, answer8_i = await agent([taskInfo, thinking7, answer7], debate_instruction, r, is_sub_task=True)
            else:
                inputs8 = [taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1]
                thinking8_i, answer8_i = await agent(inputs8, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing ΔT and selecting choice, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
            all_thinking8[r].append(thinking8_i)
            all_answer8[r].append(answer8_i)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Make final decision on closest match for ΔT.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final choice on ΔT, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs