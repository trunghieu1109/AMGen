async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = "Sub-task 1: Extract and summarize all given information about the operators P_x, P_y, P_z, the basis, and the system state vector; confirm normalization and clarify the problem requirements."
    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, extracting and summarizing given information, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent and correct summary and normalization confirmation for the problem.", is_sub_task=True)
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Compute the expectation value <P_z> in the given state by applying the P_z operator matrix to the state vector and performing the inner product. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_1 = []
    all_answer_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_1):
        thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, computing <P_z>, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        all_thinking_1_1.append(thinking1_1)
        all_answer_1_1.append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + all_thinking_1_1 + all_answer_1_1, "Sub-task 1: Final decision on <P_z> expectation value. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = "Sub-task 2: Compute the expectation value <P_z^2> by first calculating the matrix P_z^2 and then evaluating its expectation value in the given state. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_2 = []
    all_answer_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_1_2):
        thinking1_2, answer1_2 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_2, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, computing <P_z^2>, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        all_thinking_1_2.append(thinking1_2)
        all_answer_1_2.append(answer1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0] + all_thinking_1_2 + all_answer_1_2, "Sub-task 2: Final decision on <P_z^2> expectation value. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 1: Derive the uncertainty ΔP_z using the formula ΔP_z = sqrt(<P_z^2> - <P_z>^2) and interpret the result to select the correct answer choice."
    cot_sc_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1_1.content, answer1_1.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2, answer2 = await cot_sc_agents_2[i]([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, deriving uncertainty ΔP_z and interpreting result, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2] + possible_thinkings_2 + possible_answers_2, "Sub-task 3: Synthesize and choose the most consistent and correct uncertainty ΔP_z and final answer choice.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs
