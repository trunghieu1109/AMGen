async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = "Sub-task 1: Extract and summarize all given information about the crystal system, lattice parameters, angles, and the plane of interest from the query."
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
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, extracting info, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings_0 + possible_answers_0, "Sub-task 1: Synthesize and choose the most consistent extraction and summary of given information.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_1_1 = "Sub-task 1: Analyze and classify the extracted elements to understand the rhombohedral lattice geometry, the significance of equal lattice angles (30 degrees), and the Miller indices (111) plane. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
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
        agents.append(f"Debate agent {agent.id}, round 0, analyzing lattice geometry, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        all_thinking_1_1.append(thinking1_1)
        all_answer_1_1.append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo, thinking0, answer0] + all_thinking_1_1 + all_answer_1_1, "Sub-task 1: Synthesize and choose the most consistent analysis of lattice geometry and Miller indices.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Identify and clarify the appropriate formula for calculating the interplanar distance in a rhombohedral lattice, considering the lattice parameters and angles."
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking1_2, answer1_2 = await cot_sc_agents_1_2[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, identifying formula, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo, thinking0, answer0] + possible_thinkings_1_2 + possible_answers_1_2, "Sub-task 2: Synthesize and choose the most consistent formula for interplanar distance.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 1: Apply the identified formula to compute the interplanar distance of the (111) plane using the given lattice parameter (10 Angstrom) and lattice angles (30 degrees). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_2_1 = []
    all_answer_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking1_1.content, answer1_1.content, thinking1_2.content, answer1_2.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_2_1):
        thinking2_1, answer2_1 = await agent([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2], debate_instruction_2_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, computing interplanar distance, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        all_thinking_2_1.append(thinking2_1)
        all_answer_2_1.append(answer2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo, thinking1_1, answer1_1, thinking1_2, answer1_2] + all_thinking_2_1 + all_answer_2_1, "Sub-task 1: Synthesize and choose the most consistent computed interplanar distance.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_3_1 = "Sub-task 1: Evaluate the computed interplanar distance against the provided choices and prioritize to select the closest matching value. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3_1 = []
    all_answer_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking2_1.content, answer2_1.content],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_3_1):
        thinking3_1, answer3_1 = await agent([taskInfo, thinking2_1, answer2_1], debate_instruction_3_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, evaluating choices, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        all_thinking_3_1.append(thinking3_1)
        all_answer_3_1.append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2_1, answer2_1] + all_thinking_3_1 + all_answer_3_1, "Sub-task 1: Synthesize and select the closest matching interplanar distance from the given choices.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_1, answer3_1, sub_tasks, agents)
    return final_answer, logs
