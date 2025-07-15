async def forward_156(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize the outbreak context and retroviral genome characteristics from the user query."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, classify diagnostic targets and modalities (viral RNA/cDNA vs. host antibodies), noting pros and cons for rapid early infection detection."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        t2, a2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {t2.content}; answer: {a2.content}")
        possible_thinkings.append(t2)
        possible_answers.append(a2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
                                                    "Sub-task 2: Synthesize and choose the most consistent classification for diagnostic targets and modalities.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc3_instruction = "Sub-task 3: Design and outline candidate assay workflows: RT-PCR, nested PCR, real-time qPCR, and ELISA formats with key reagents and detection steps."
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc3_instruction, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        t3, a3 = await cot_agents[i]([taskInfo, thinking2, answer2], cot_sc3_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {t3.content}; answer: {a3.content}")
        possible_thinkings3.append(t3)
        possible_answers3.append(a3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
                                                    "Sub-task 3: Synthesize and select the most consistent assay designs.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Evaluate each candidate assay for turnaround time, sensitivity, specificity, equipment needs, and contamination risk." + reflect_inst
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Reflexion"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    N_max = self.max_round
    for i in range(N_max):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent4([taskInfo, thinking4, answer4], critic_inst, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3, thinking4, answer4, feedback], cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = "Sub-task 5: Debate and prioritize the optimal assay workflow (e.g., RT-qPCR on cDNA) for rapid, accurate field deployment." + debate_instr
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Debate"}
    all_thinking5 = [[] for _ in range(N_max)]
    all_answer5 = [[] for _ in range(N_max)]
    for r in range(N_max):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                t5, a5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, is_sub_task=True)
            else:
                t5, a5 = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction5, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t5.content}; answer: {a5.content}")
            all_thinking5[r].append(t5)
            all_answer5[r].append(a5)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Final synthesis and prioritization." + final_instr, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs