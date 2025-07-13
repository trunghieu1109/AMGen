async def forward_29(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalization and Assumptions

    # Subtask 1: Formal definition of grid, chips, placement rules (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the grid, chip types, and placement rules, including the meaning of 'each cell contains at most one chip' "
        "and the color uniformity constraints for rows and columns. Explicitly state that chips are indistinguishable and clarify how this affects counting."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing problem, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Clarify assumptions about empty rows/columns (SC_CoT)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, clarify and formalize assumptions about empty rows and columns: "
        "whether they have assigned colors or are considered neutral, and how this neutrality affects the uniformity and maximality conditions. "
        "Avoid ambiguous interpretations by explicitly stating these assumptions."
    )
    N_sc_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_2):
        thinking_i, answer_i = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, clarifying empty rows/columns, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_2.append(answer_i)
        possible_thinkings_0_2.append(thinking_i)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent assumptions about empty rows/columns.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Precisely define maximality condition (SC_CoT)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Precisely define the maximality condition for chip placement: what it means for a placement to be maximal in terms of adding chips without violating row or column color uniformity. "
        "Specify how maximality applies to empty cells and unchosen rows or columns, ensuring no ambiguity."
    )
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking_i, answer_i = await cot_agents_0_3[i]([taskInfo, thinking_0_1, thinking_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, defining maximality, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_3.append(answer_i)
        possible_thinkings_0_3.append(thinking_i)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent maximality definition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Compatibility and Structure Characterization

    # Subtask 1: Prove necessary and sufficient conditions on row/column color assignments (Reflexion)
    reflect_inst_1_1 = (
        "Sub-task 1: Identify and prove the necessary and sufficient conditions on row and column color assignments for a placement to be valid, "
        "focusing on the compatibility of colors at intersections of occupied rows and columns. Emphasize that occupied cells must have matching row and column colors. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_1_1 = [taskInfo, thinking_0_3]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflect_inst_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, reflect_inst_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, proving compatibility, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1_1([taskInfo, thinking_1_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, feedback])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, reflect_inst_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining compatibility proof, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Characterize structure of maximal placements (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Characterize the structure of maximal placements in terms of subsets of rows and columns chosen to be occupied and their color assignments. "
        "Explicitly incorporate the maximality condition by ensuring that no additional row or column can be added without violating uniformity."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, characterizing maximal placements, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent characterization of maximal placements.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: Counting Problem and Validation

    # Subtask 1: Translate structure into combinatorial counting problem (Debate)
    debate_instr_2_1 = (
        "Sub-task 1: Translate the structural characterization of maximal placements into a combinatorial counting problem. "
        "Express the number of valid configurations as a function of chosen subsets of rows and columns and their color assignments, explicitly including maximality constraints to prevent overcounting. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_1_2], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2] + all_thinking_2_1[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_2_1[r].append(thinking_i)
            all_answer_2_1[r].append(answer_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide an updated counting formulation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Simplify counting expression with explicit maximality checks (Reflexion)
    reflect_inst_2_2 = (
        "Sub-task 2: Simplify the combinatorial counting expression by exploiting symmetries, indistinguishability of chips, and maximality constraints. "
        "Break down the counting into modular parts: (a) enumeration of subsets and color assignments, (b) maximality verification for each configuration, and (c) summation of valid counts. "
        "Avoid assumptions that bypass maximality enforcement. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflect_inst_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, simplifying counting with maximality, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflect_inst_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining counting simplification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Subtask 3: Implement explicit maximality checks for small grids (SC_CoT)
    cot_sc_instruction_2_3 = (
        "Sub-task 3: Implement explicit maximality checks for small grid cases (2x2, 3x3) by direct enumeration or algorithmic verification. "
        "Use these results to validate and refine the counting formula before generalizing to the 5x5 grid. Document findings and adjustments needed to ensure correctness."
    )
    N_sc_2_3 = self.max_sc
    cot_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_3)]
    possible_answers_2_3 = []
    possible_thinkings_2_3 = []
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_sc_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_3):
        thinking_i, answer_i = await cot_agents_2_3[i]([taskInfo, thinking_2_2], cot_sc_instruction_2_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_3[i].id}, maximality checks small grids, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_3.append(answer_i)
        possible_thinkings_2_3.append(thinking_i)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + possible_thinkings_2_3, "Sub-task 3: Synthesize and validate counting formula with maximality checks on small grids.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Stage 3: Final Aggregation and Cross-Validation

    # Subtask 1: Aggregate counts and compute final total (SC_CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Aggregate the simplified combinatorial components and compute the final total number of maximal valid placements on the 5x5 grid. "
        "Verify that the result satisfies all constraints, especially maximality, by cross-validating with enumerations and maximality checks from previous subtasks."
    )
    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_1):
        thinking_i, answer_i = await cot_agents_3_1[i]([taskInfo, thinking_2_3], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, aggregating final count, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_3_1.append(answer_i)
        possible_thinkings_3_1.append(thinking_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_thinkings_3_1, "Sub-task 1: Synthesize and compute final total count with maximality enforcement.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    # Subtask 2: Cross-validate numeric results and reconcile discrepancies (Debate)
    debate_instr_3_2 = (
        "Sub-task 2: Cross-validate numeric results obtained from different agents. If discrepancies arise, perform a reconciliation subtask focused on analyzing and resolving differences through enumeration, combinatorial proof, or maximality enforcement checks. "
        "Ensure the final numeric answer is deterministic and rigorously justified. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instr_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_3_1], debate_instr_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1] + all_thinking_3_2[r-1]
                thinking_i, answer_i = await agent(input_infos, debate_instr_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-validating results, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_2[r].append(thinking_i)
            all_answer_3_2[r].append(answer_i)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final reconciled numeric answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
