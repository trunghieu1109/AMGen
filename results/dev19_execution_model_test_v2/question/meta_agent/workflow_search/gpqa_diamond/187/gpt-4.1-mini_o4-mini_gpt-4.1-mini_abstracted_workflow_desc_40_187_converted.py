async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Extract and summarize the defining features of the rhombohedral crystal from the query, "
        "including lattice parameter, lattice angles, and the plane indices (111)."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting and summarizing crystal features, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instr_1 = (
        "Sub-task 1: Analyze and classify the geometric and crystallographic elements relevant to the rhombohedral lattice, "
        "including the relationship between lattice parameters, angles, and Miller indices for the (111) plane. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_1, answer_1 = await agent([taskInfo, thinking_0, answer_0], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking_0, answer_0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking_1, answer_1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing geometric and crystallographic elements, thinking: {thinking_1.content}; answer: {answer_1.content}")
            all_thinking_1[r].append(thinking_1)
            all_answer_1[r].append(answer_1)
    final_decision_instr_1 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo, thinking_0, answer_0] + all_thinking_1[-1] + all_answer_1[-1],
        "Sub-task 1: Analyze and classify geometric and crystallographic elements." + final_decision_instr_1,
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 1: Compute the formula for the interplanar spacing d_(hkl) in a rhombohedral lattice using the lattice parameter and angles, specifically for the (111) plane, "
        "based on the analysis from previous stage."
    )
    N_sc_2 = self.max_sc
    cot_agents_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_2)
    ]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking_1, answer_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, computing formula for interplanar spacing, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2(
        [taskInfo, thinking_1, answer_1] + possible_thinkings_2 + possible_answers_2,
        "Sub-task 2: Synthesize and choose the most consistent formula for interplanar spacing.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 1: Apply the derived formula and given numerical values (lattice parameter = 10 Angstrom, angles = 30 degrees) to calculate the interplanar distance of the (111) plane."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc_3)
    ]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", thinking_2, answer_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_2, answer_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, applying formula with numerical values, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3(
        [taskInfo, thinking_2, answer_2] + possible_thinkings_3 + possible_answers_3,
        "Sub-task 3: Synthesize and choose the most consistent calculated interplanar distance.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4 = (
        "Sub-task 1: Evaluate the calculated interplanar distance against the provided multiple-choice options to identify the closest match. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    subtask_desc_4 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instr_4,
        "context": ["user query", thinking_3, answer_3],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking_4, answer_4 = await agent([taskInfo, thinking_3, answer_3], debate_instr_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking_3, answer_3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking_4, answer_4 = await agent(input_infos_4, debate_instr_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating against choices, thinking: {thinking_4.content}; answer: {answer_4.content}")
            all_thinking_4[r].append(thinking_4)
            all_answer_4[r].append(answer_4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await final_decision_agent_4(
        [taskInfo, thinking_3, answer_3] + all_thinking_4[-1] + all_answer_4[-1],
        "Sub-task 4: Evaluate and select the closest matching interplanar distance.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst_5 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_5 = (
        "Sub-task 1: Combine the quantitative calculation and evaluation results to select and justify the best answer choice for the interplanar distance of the (111) plane. "
        + reflect_inst_5
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking_3, answer_3, thinking_4, answer_4]
    subtask_desc_5 = {
        "subtask_id": "stage_5.subtask_1",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking_3, answer_3, thinking_4, answer_4],
        "agent_collaboration": "Reflexion"
    }
    thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, combining calculation and evaluation, thinking: {thinking_5.content}; answer: {answer_5.content}")
    critic_inst_5 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    )
    for i in range(N_max_5):
        feedback_5, correct_5 = await critic_agent_5([taskInfo, thinking_5, answer_5], critic_inst_5, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback_5.content}; answer: {correct_5.content}")
        if correct_5.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking_5, answer_5, feedback_5])
        thinking_5, answer_5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining answer, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Stage 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
