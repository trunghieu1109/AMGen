async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Parse the original decay 2A → 2B + 2E + 2V: list all emitted particles, note that the E‐particle energy spectrum is continuous, and record the endpoint value Q."
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    sub1_desc = {"subtask_id": "subtask_1", "instruction": cot_instruction1, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_instruction1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, parsing original decay, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    sub1_desc["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(sub1_desc)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Parse the variant decay 2A → 2B + 2E + M: identify the change in emitted particles (two V replaced by one massless M) and highlight the difference in total mass of the undetected system."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    sub2_desc = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, parsing variant decay, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2.content)
        thinking_map2[answer2.content] = thinking2
        answer_map2[answer2.content] = answer2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[answer2_content]
    answer2 = answer_map2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    sub2_desc["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(sub2_desc)
    print("Step 2: ", sub_tasks[-1])
    cot_instruction3 = "Sub-task 3: Explicitly define the mass parameters (m_V > 0, m_M = 0) and state assumptions about masses of A, B, and E to ensure unambiguous energy-balance context."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    sub3_desc = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, defining mass parameters, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    sub3_desc["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(sub3_desc)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Identify and summarize the governing physical principles—energy and momentum conservation plus phase-space volume considerations—that determine spectrum continuity and endpoint energy."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    sub4_desc = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, summarizing physical principles, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    sub4_desc["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(sub4_desc)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_instruction5 = "Sub-task 5: Perform a quantitative endpoint comparison: derive the expression ΔQ = Q_variant − Q_original = 2(m_V − m_M)c^2 to show how replacing two massive V particles with a massless M shifts the available kinetic energy for E."
    cot_agents5 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers5 = []
    thinking_map5 = {}
    answer_map5 = {}
    sub5_desc = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5a, answer5a = await cot_agents5[i]([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, deriving ΔQ expression, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers5.append(answer5a.content)
        thinking_map5[answer5a.content] = thinking5a
        answer_map5[answer5a.content] = answer5a
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinking_map5[answer5_content]
    answer5 = answer_map5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    sub5_desc["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(sub5_desc)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction6 = "Sub-task 6: Using phase-space arguments, determine whether the total E spectrum in the variant decay remains continuous or becomes discrete given the emission of a single massless particle M instead of two massive V’s."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    sub6_desc = {"subtask_id": "subtask_6", "instruction": cot_instruction6, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, determining spectrum continuity, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    sub6_desc["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(sub6_desc)
    print("Step 6: ", sub_tasks[-1])
    cot_reflect_instruction7 = "Sub-task 7: Conduct a self-consistency check and brief reflection: verify the sign of ΔQ and the conclusion about spectrum continuity, checking for any logical or algebraic errors."
    cot_agent7 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    sub7_desc = {"subtask_id": "subtask_7", "instruction": cot_reflect_instruction7, "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "Reflexion"}
    inputs7 = [taskInfo, thinking5, answer5, thinking6, answer6]
    thinking7, answer7 = await cot_agent7(inputs7, cot_reflect_instruction7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent7.id}, initial verification, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(self.max_round):
        feedback7, correct7 = await critic_agent7([taskInfo, thinking7, answer7], "Please review the verification of ΔQ sign and spectrum continuity and provide feedback if incorrect.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent7.id}, feedback round {i}, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        inputs7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent7(inputs7, cot_reflect_instruction7, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent7.id}, refined verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    sub7_desc["response"] = {"thinking": thinking7, "answer": answer7}
    logs.append(sub7_desc)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction8 = "Sub-task 8: Map the validated conclusions (continuous vs. discrete spectrum and endpoint increase vs. decrease) to the correct answer choice (A, B, C, or D) and output only that letter."
    cot_agent8 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    sub8_desc = {"subtask_id": "subtask_8", "instruction": cot_instruction8, "context": ["user query", "thinking of subtask 7", "answer of subtask 7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, mapping to answer choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    sub8_desc["response"] = {"thinking": thinking8, "answer": answer8}
    logs.append(sub8_desc)
    print("Step 8: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs