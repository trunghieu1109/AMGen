async def forward_182(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    sccot_instr = "Sub-task 1: Extract and summarize the substrate’s structural features—ring size, position and count of C=C bonds, formyl and carboxyl groups—and compute its initial degrees of unsaturation. Embed feedback to avoid rushing mechanistic assumptions."
    sccot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings1, possible_answers1 = [], []
    subtask_desc1 = {"subtask_id": "stage1_subtask_1", "instruction": sccot_instr, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in sccot_agents:
        thinking1_i, answer1_i = await agent([taskInfo], sccot_instr, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing substrate features, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent answer for initial IHD.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2: Survey known transformations effected by red phosphorus and excess HI, including reduction of aldehydes and acids, decarboxylation, and hydrogenation or cleavage of C=C bonds. Explicitly challenge the assumption that alkene bonds remain untouched by citing literature precedents." + reflect_inst
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs2 = [taskInfo, thinking1, answer1]
    subtask_desc2 = {"subtask_id": "stage1_subtask_2", "instruction": cot_reflect_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "Reflexion"}
    thinking2, answer2 = await cot_agent2(inputs2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, analyzing red P/HI transformations, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        inputs2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent2(inputs2, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refinement thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction = "Sub-task 3: Generate and debate all plausible reaction pathways based on outcomes from Subtask 2. For each pathway, propose a candidate product structure and note its implications for unsaturation (rings and double bonds). Ensure alternative mechanisms (e.g., partial vs. full hydrogenation) are considered. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "stage1_subtask_3", "instruction": debate_instruction, "context": ["user query", "thinking2", "answer2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs3, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    sccot_instr4 = "Sub-task 4: Select the most likely final product structure by weighing mechanistic plausibility and experimental precedent. Validate that this choice corrects the earlier error of overestimating unsaturation and document the rationale."
    sccot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings4, possible_answers4 = [], []
    subtask_desc4 = {"subtask_id": "stage1_subtask_4", "instruction": sccot_instr4, "context": ["user query", "thinking3", "answer3"], "agent_collaboration": "SC_CoT"}
    for agent in sccot_agents4:
        thinking4_i, answer4_i = await agent([taskInfo, thinking2, answer2, thinking3, answer3], sccot_instr4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, selecting product, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent product structure.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_instr5 = "Sub-task 5: Determine the molecular formula of the selected product and calculate its index of hydrogen deficiency (IHD), ensuring calculations reflect the validated structure from Stage 1. Avoid repeating prior miscounts by double-checking each unsaturation source."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "stage2_subtask_1", "instruction": cot_instr5, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instr5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, calculating molecular formula and IHD, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instr6 = "Sub-task 6: Compare the computed IHD with the provided choices (1, 3, 0, 5) and select the correct answer, documenting the match clearly."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "stage2_subtask_2", "instruction": cot_instr6, "context": ["user query", "thinking5", "answer5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instr6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, selecting the correct IHD, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs