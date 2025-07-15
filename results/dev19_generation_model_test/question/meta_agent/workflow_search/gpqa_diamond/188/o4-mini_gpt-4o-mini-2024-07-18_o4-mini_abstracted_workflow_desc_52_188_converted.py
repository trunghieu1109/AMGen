async def forward_188(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction = "Sub-task 1: For each choice (Magnon, Skyrmion, Pion, Phonon), extract and summarize the spontaneously broken symmetry that produces it, if any."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc1 = {"subtask_id": "stage1_subtask1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_i, answer_i = await cot_agents[i]([taskInfo], cot_sc_instruction, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings_1.append(thinking_i)
        possible_answers_1.append(answer_i)
    final_instr1 = "Given the above reasoning and summaries, synthesize the most consistent and correct mappings between each particle and its spontaneously broken symmetry."
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent summary for each particle." + final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 2: Based on the summary from Sub-task 1, categorize each particle as a true Goldstone mode, pseudo–Goldstone mode, or topological soliton." + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max2)]
    all_answer2 = [[] for _ in range(N_max2)]
    subtask_desc2 = {"subtask_id": "stage1_subtask2", "instruction": debate_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Debate"}
    for r in range(N_max2):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2_r, answer2_r = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                thinking2_r, answer2_r = await agent([taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_r.content}; answer: {answer2_r.content}")
            all_thinking2[r].append(thinking2_r)
            all_answer2[r].append(answer2_r)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final categorization."
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Finalize categorization of each particle." + final_instr2, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Assess whether pseudo–Goldstone modes (e.g., pions) count as 'associated with spontaneous symmetry breaking' or require special treatment. " + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max3 = self.max_round
    cot_inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "stage2_subtask1", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instr4 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction4 = "Sub-task 4: Determine which of the four particles is not associated with any spontaneously-broken symmetry, using the classifications and assessments from earlier subtasks." + debate_instr4
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    subtask_desc4 = {"subtask_id": "stage3_subtask1", "instruction": debate_instruction4, "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_r, answer4_r = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                thinking4_r, answer4_r = await agent([taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1], debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_r.content}; answer: {answer4_r.content}")
            all_thinking4[r].append(thinking4_r)
            all_answer4[r].append(answer4_r)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Determine which particle is not associated with a spontaneously broken symmetry." + final_instr4, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs