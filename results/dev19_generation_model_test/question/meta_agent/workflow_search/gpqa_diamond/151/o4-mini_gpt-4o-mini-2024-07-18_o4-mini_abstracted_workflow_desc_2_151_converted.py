async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT
    sc_cot_instruction1 = "Sub-task 1: Extract and summarize key experimental details from the query including organisms, peptide, phenotype, and assay."
    N1 = self.max_sc
    sc_cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1, possible_answers1 = [], []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": sc_cot_instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking, answer = await sc_cot_agents1[i]([taskInfo], sc_cot_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents1[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent summary for key experimental details.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    # Sub-task 2: CoT
    cot_instruction2 = "Sub-task 2: Analyze relationships among quorum-sensing peptide treatment, shmoo formation, transcriptional activation, and active chromatin immunoprecipitation."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])

    # Sub-task 3: SC_CoT
    sc_cot_instruction3 = "Sub-task 3: Characterize the four candidate protein complexes (pre-initiation, pre-replication, enhancer, nucleosome histone) in terms of function, timing, and association with active chromatin."
    N3 = self.max_sc
    sc_cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3, possible_answers3 = [], []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": sc_cot_instruction3, "context": ["user query", "thinking2", "answer2"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking, answer = await sc_cot_agents3[i]([taskInfo, thinking2, answer2], sc_cot_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_cot_agents3[i].id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent characterization of the four complexes.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])

    # Sub-task 4: Debate
    debate_instr4 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated evaluation."
    debate_instruction4 = "Sub-task 4: Compare and evaluate which complex is least likely to be enriched in an active-chromatin ChIP-MS assay during transcriptional activation in the shmoo." + debate_instr4
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N4 = self.max_round
    all_thinking4 = [[] for _ in range(N4)]
    all_answer4 = [[] for _ in range(N4)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction4, "context": ["user query", "thinking2", "answer2", "thinking3", "answer3"], "agent_collaboration": "Debate"}
    for r in range(N4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking_r, answer_r = await agent([taskInfo, thinking2, answer2, thinking3, answer3], debate_instruction4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking_r, answer_r = await agent(inputs, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_r.content}; answer: {answer_r.content}")
            all_thinking4[r].append(thinking_r)
            all_answer4[r].append(answer_r)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])

    # Sub-task 5: Reflexion
    reflect_inst5 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction5 = "Sub-task 5: Formulate and present the final conclusion identifying the complex whose proteins will be least observed, with supporting rationale. " + reflect_inst5
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N5 = self.max_round
    cot_inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5, "context": ["user query", "thinking4", "answer4"], "agent_collaboration": "Reflexion"}
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N5):
        feedback, correct = await critic_agent5([taskInfo, thinking5, answer5],
            "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
            i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs