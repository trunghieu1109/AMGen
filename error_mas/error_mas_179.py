async def forward_179(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and summarize given physical parameters using Debate
    debate_instr_0 = "Sub-task 1: Extract and summarize all given physical parameters, constants, and constraints from the problem statement." + \
                    "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, extracting parameters, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], 
                                                      "Sub-task 1: Extract and summarize parameters." + 
                                                      "Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                      is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: Calculate energy contributions using SC_CoT
    # Subtask 1: Central charge interactions
    cot_sc_instruction_1 = "Sub-task 1: Calculate the electrostatic potential energy contributions from the interaction between the central charge and each of the 12 charges on the sphere." + \
                         "Consider multiple possible approaches and configurations to ensure accuracy."
    N_sc_1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking_1_1, answer_1_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, calculating central charge interactions, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1.append(answer_1_1)
        possible_thinkings_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0, answer_0] + possible_thinkings_1 + possible_answers_1, 
                                                            "Sub-task 1: Synthesize and choose the most consistent answer for central charge interactions.", 
                                                            is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Mutual interactions among 12 charges
    cot_sc_instruction_2 = "Sub-task 2: Calculate the electrostatic potential energy contributions from the interactions among the 12 charges constrained on the sphere, considering their minimum energy configuration." + \
                         "Consider multiple possible configurations and approaches to find minimal energy." 
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_1_2, answer_1_2 = await cot_agents_2[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating mutual interactions, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_2.append(answer_1_2)
        possible_thinkings_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0, answer_0] + possible_thinkings_2 + possible_answers_2, 
                                                            "Sub-task 2: Synthesize and choose the most consistent answer for mutual interactions among 12 charges.", 
                                                            is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: Combine energy contributions using Reflexion
    reflect_inst_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 1: Combine the energy contributions from the central charge interactions and the 12 charges' mutual interactions to compute the total minimum electrostatic potential energy of the system." + reflect_inst_2
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2 = self.max_round
    cot_inputs_2 = [taskInfo, thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2,
        "context": ["user query", thinking_0, answer_0, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, combining energies, thinking: {thinking_2.content}; answer: {answer_2.content}")
    for i in range(N_max_2):
        feedback_2, correct_2 = await critic_agent_2([taskInfo, thinking_2, answer_2], 
                                                    "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                                    i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, providing feedback, thinking: {feedback_2.content}; answer: {correct_2.content}")
        if correct_2.content == "True":
            break
        cot_inputs_2.extend([thinking_2, answer_2, feedback_2])
        thinking_2, answer_2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refining combined energy, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Compare computed energy with choices using Debate
    debate_instr_3 = "Sub-task 1: Compare the computed total minimum energy with the provided answer choices and select the one that matches the calculated value to three decimal places." + \
                    "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2, answer_2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2, answer_2] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with choices, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo, thinking_2, answer_2] + all_thinking_3[-1] + all_answer_3[-1], 
                                                      "Sub-task 1: Final decision on the best matching answer choice." + 
                                                      "Given all the above thinking and answers, reason over them carefully and provide a final answer.", 
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
