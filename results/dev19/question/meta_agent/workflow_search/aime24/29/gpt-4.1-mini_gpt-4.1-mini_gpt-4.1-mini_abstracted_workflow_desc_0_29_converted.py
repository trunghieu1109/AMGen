async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Constraint Analysis and Synthesis

    # Sub-task 1: Analyze 'each cell contains at most one chip' with SC_CoT
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Analyze and clearly state the implications of the condition that each cell contains at most one chip in the 5x5 grid. "
        "Emphasize that this limits chip placement to at most one chip per cell and that the total number of chips placed cannot exceed 25. "
        "Avoid assuming that all chips must be placed."
    )
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, analyzing cell occupancy, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent answer for cell occupancy constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Sub-task 2: Analyze 'all chips in the same row have the same color' with SC_CoT
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Analyze and clearly state the implications of the condition that all chips in the same row have the same color. "
        "Emphasize that each row is monochromatic if it contains any chips, and that the row color assignment is a choice between white or black. "
        "Avoid imposing constraints linking row colors to column colors at this stage."
    )
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_2[i]([taskInfo], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing row color uniformity, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent answer for row color constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Sub-task 3: Analyze 'all chips in the same column have the same color' with SC_CoT
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Analyze and clearly state the implications of the condition that all chips in the same column have the same color. "
        "Emphasize that each column is monochromatic if it contains any chips, and that the column color assignment is a choice between white or black. "
        "Avoid imposing constraints linking column colors to row colors at this stage."
    )
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_3[i]([taskInfo], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, analyzing column color uniformity, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent answer for column color constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Sub-task 4: Analyze maximality condition with Debate
    debate_instr_0_4 = (
        "Sub-task 4: Analyze and clearly define the maximality condition: that adding any additional chip would violate either the uniform color condition in a row or column or the one-chip-per-cell condition. "
        "Explicitly clarify that maximality means every cell where the row and column colors agree must be occupied by a chip, and no chips can be added to cells where the row and column colors differ. "
        "Avoid introducing additional constraints on the sums of row and column color assignments. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_0_4 = self.max_round
    all_thinking_0_4 = [[] for _ in range(N_max_0_4)]
    all_answer_0_4 = [[] for _ in range(N_max_0_4)]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0_subtask_4",
        "instruction": debate_instr_0_4,
        "context": ["user query", thinking_0_1, thinking_0_2, thinking_0_3],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_4):
        for i, agent in enumerate(debate_agents_0_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3], debate_instr_0_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1, thinking_0_2, thinking_0_3] + all_thinking_0_4[r-1]
                thinking, answer = await agent(input_infos, debate_instr_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing maximality, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_4[r].append(thinking)
            all_answer_0_4[r].append(answer)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3] + all_thinking_0_4[-1], "Sub-task 4: Synthesize and finalize maximality condition analysis. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    # Sub-task 5: Synthesize combined constraints with SC_CoT
    cot_sc_instruction_0_5 = (
        "Sub-task 5: Synthesize the combined constraints from subtasks 1-4 to characterize the structure of valid chip placements. "
        "Explicitly incorporate chip availability limits (25 white and 25 black chips as upper bounds) and the maximality condition as saturation of compatible cells only. "
        "Verify logical consistency between chip availability, grid capacity, and maximality, and avoid contradictory constraints such as requiring simultaneous full placement of all chips."
    )
    cot_agents_0_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_5 = []
    possible_thinkings_0_5 = []
    subtask_desc_0_5 = {
        "subtask_id": "stage_0_subtask_5",
        "instruction": cot_sc_instruction_0_5,
        "context": ["user query", thinking_0_1, thinking_0_2, thinking_0_3, thinking_0_4],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_5[i]([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3, thinking_0_4], cot_sc_instruction_0_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_5[i].id}, synthesizing combined constraints, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_5.append(answer)
        possible_thinkings_0_5.append(thinking)
    final_decision_agent_0_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_5, answer_0_5 = await final_decision_agent_0_5([taskInfo] + possible_thinkings_0_5, "Sub-task 5: Synthesize and choose the most consistent combined constraints characterization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_0_5.content}; answer - {answer_0_5.content}")
    subtask_desc_0_5['response'] = {"thinking": thinking_0_5, "answer": answer_0_5}
    logs.append(subtask_desc_0_5)
    print("Step 0.5: ", sub_tasks[-1])

    # Stage 1: Enumerate Row and Column Color Assignments

    # Sub-task 1: Enumerate all possible row color assignments with SC_CoT
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible color assignments to the 5 rows, where each row is assigned either white or black independently. "
        "Emphasize that no additional constraints from maximality or chip availability restrict these assignments at this stage."
    )
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_5],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_1_1[i]([taskInfo, thinking_0_5], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating row colors, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_5] + possible_thinkings_1_1, "Sub-task 1: Synthesize and finalize enumeration of row color assignments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Sub-task 2: Enumerate all possible column color assignments with SC_CoT
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Enumerate all possible color assignments to the 5 columns, where each column is assigned either white or black independently. "
        "Emphasize that no additional constraints from maximality or chip availability restrict these assignments at this stage."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_5],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_1_2[i]([taskInfo, thinking_0_5], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, enumerating column colors, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0_5] + possible_thinkings_1_2, "Sub-task 2: Synthesize and finalize enumeration of column color assignments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Sub-task 3: Determine cells where row and column colors agree with Debate
    debate_instr_1_3 = (
        "Sub-task 3: Determine the set of cells where the row and column colors agree for each pair of row and column color assignments. "
        "Clarify that these cells are exactly those where chips must be placed to satisfy maximality. "
        "Avoid imposing constraints that limit the number of such cells beyond chip availability limits. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": debate_instr_1_3,
        "context": ["user query", thinking_1_1, thinking_1_2, thinking_0_5],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_1, thinking_1_2, thinking_0_5], debate_instr_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1, thinking_1_2, thinking_0_5] + all_thinking_1_3[r-1]
                thinking, answer = await agent(input_infos, debate_instr_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining agreement cells, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_3[r].append(thinking)
            all_answer_1_3[r].append(answer)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo, thinking_1_1, thinking_1_2, thinking_0_5] + all_thinking_1_3[-1], "Sub-task 3: Synthesize and finalize determination of agreement cells. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Sub-task 4: Verify chip availability constraints with SC_CoT
    cot_sc_instruction_1_4 = (
        "Sub-task 4: For each pair of row and column color assignments, verify that the number of cells where colors agree does not exceed the available chips of that color (white or black). "
        "Since the grid size and chip availability are equal (25 each), confirm that this condition is always satisfied. "
        "Avoid introducing invalid constraints linking row and column color counts."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1_subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_1_4[i]([taskInfo, thinking_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, verifying chip availability, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_4.append(answer)
        possible_thinkings_1_4.append(thinking)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 4: Synthesize and finalize chip availability verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Counting and Verification

    # Sub-task 1: Count total valid maximal configurations with SC_CoT
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Combine all possible row and column color assignments (2^5 each) to count the total number of valid maximal chip placement configurations on the 5x5 grid. "
        "Use the fact that maximality requires placing chips in all cells where row and column colors agree, and that chip availability constraints are not violated. "
        "Avoid excluding any configurations based on incorrect assumptions."
    )
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_4],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_2_1[i]([taskInfo, thinking_1_4], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, counting valid configurations, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_1.append(answer)
        possible_thinkings_2_1.append(thinking)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Synthesize and finalize counting of valid maximal configurations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2: Verify total count respects chip availability and maximality with Debate
    debate_instr_2_2 = (
        "Sub-task 2: Verify that the total count of configurations respects the chip availability limits (25 white and 25 black chips as upper bounds) and the maximality condition. "
        "Confirm that no contradictions arise and that the final count matches the expected 2^{10} = 1024 configurations. "
        "If contradictions are found, prompt revision of assumptions or earlier subtasks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_2 = self.max_round
    all_thinking_2_2 = [[] for _ in range(N_max_2_2)]
    all_answer_2_2 = [[] for _ in range(N_max_2_2)]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": debate_instr_2_2,
        "context": ["user query", thinking_2_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_2):
        for i, agent in enumerate(debate_agents_2_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_1], debate_instr_2_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1] + all_thinking_2_2[r-1]
                thinking, answer = await agent(input_infos, debate_instr_2_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying final count, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_2[r].append(thinking)
            all_answer_2_2[r].append(answer)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo, thinking_2_1] + all_thinking_2_2[-1], "Sub-task 2: Synthesize and finalize verification of total count.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
