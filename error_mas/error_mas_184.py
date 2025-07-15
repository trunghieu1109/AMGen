async def forward_184(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0 = "Sub-task 1: Extract and summarize the defining features of the Hamiltonian operator H = epsilon sigma dot n, including properties of sigma (Pauli matrices), unit vector n, and constant epsilon." + \
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_0 = []
    all_answer_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0):
        thinking, answer = await agent([taskInfo], debate_instr_0, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, analyzing Hamiltonian features, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_0.append(thinking)
        all_answer_0.append(answer)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + all_thinking_0 + all_answer_0, 
                                                    "Sub-task 1: Synthesize and choose the most consistent summary of Hamiltonian features." + \
                                                    " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1: Apply the known spectral properties of the operator sigma dot n to determine its eigenvalues and eigenvectors, considering sigma dot n is dimensionless and has eigenvalues ±1." 
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1_1, answer1_1 = await cot_agents_1_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, spectral properties of sigma dot n, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers_1_1.append(answer1_1)
        possible_thinkings_1_1.append(thinking1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + possible_thinkings_1_1 + possible_answers_1_1, 
                                                          "Sub-task 1: Synthesize and choose the most consistent eigenvalues of sigma dot n.",
                                                          is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instr_1_2 = "Sub-task 2: Analyze the physical interpretation of the Hamiltonian eigenvalues, clarifying the role of hbar/2 factors and whether the Pauli matrices include spin angular momentum scaling." + \
                      " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_2 = []
    all_answer_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instr_1_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_2):
        thinking1_2, answer1_2 = await agent([taskInfo, thinking0, answer0, thinking1_1, answer1_1], debate_instr_1_2, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, physical interpretation of eigenvalues, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        all_thinking_1_2.append(thinking1_2)
        all_answer_1_2.append(answer1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0, thinking1_1, answer1_1] + all_thinking_1_2 + all_answer_1_2, 
                                                          "Sub-task 2: Synthesize and choose the most consistent physical interpretation of eigenvalues." + \
                                                          " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                          is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 1: Integrate the mathematical eigenvalues of sigma dot n with the physical constants epsilon and hbar to produce the final form of the Hamiltonian eigenvalues." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2 = [taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2]
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking1_1, answer1_1, thinking1_2, answer1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, integrating eigenvalues and physical constants, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        critic_inst_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], critic_inst_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining integration, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = "Sub-task 1: Evaluate the multiple-choice options against the derived eigenvalues and prioritize the correct answer based on dimensional consistency and physical interpretation." + \
                    " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3 = []
    all_answer_3 = []
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3):
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr_3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, evaluating multiple-choice options, thinking: {thinking3.content}; answer: {answer3.content}")
        all_thinking_3.append(thinking3)
        all_answer_3.append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking_3 + all_answer_3, 
                                                    "Sub-task 1: Final evaluation and selection of correct eigenvalues from multiple-choice options." + \
                                                    " Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
