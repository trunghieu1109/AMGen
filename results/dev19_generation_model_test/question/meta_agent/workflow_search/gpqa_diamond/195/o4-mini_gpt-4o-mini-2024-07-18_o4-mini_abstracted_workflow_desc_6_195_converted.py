async def forward_195(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0 - Sub-task 1: Extract and summarize inputs (SC_CoT)
    cot_sc_instruction1 = "Sub-task 1: Extract and summarize all given physical quantities, relationships, and the four candidate expressions for v_max from the query."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1, possible_answers1 = [], []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, extracting inputs, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent summary.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 0 - Sub-task 2: Identify and classify physics principles (SC_CoT)
    cot_sc_instruction2 = "Sub-task 2: Identify and classify relevant physics principles: Hooke’s law, relativistic kinetic energy, and energy conservation based on the summary from Sub-task 1."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2, possible_answers2 = [], []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", thinking1, answer1], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, classifying principles, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent classification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 1 - Sub-task 3: Derive expression for v_max (SC_CoT)
    cot_sc_instruction3 = "Sub-task 3: Derive an expression for v_max by equating ½ k A^2 to the relativistic kinetic energy (γ−1) m c^2 and solving for v."
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3, possible_answers3 = [], []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", thinking2, answer2], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, deriving v_max, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent derivation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 2 - Sub-task 4: Simplify and match to given choices (SC_CoT)
    cot_sc_instruction4 = "Sub-task 4: Simplify the derived expression for v_max and match it to one of the provided candidate formulas."
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4, possible_answers4 = [], []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query", thinking3, answer3], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, simplifying and matching, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_agent4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent match.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 2 - Sub-task 5: Verify consistency (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction5 = "Sub-task 5: Verify the physical consistency of the selected formula by checking v_max < c and domain constraints on kA^2 relative to m c^2." + reflect_inst
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5, "context": ["user query", thinking4, answer4], "agent_collaboration": "Reflexion"}
    cot_inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial consistency check, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct?: {correct5.content}")
        if correct5.content.strip() == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refined consistency check, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs