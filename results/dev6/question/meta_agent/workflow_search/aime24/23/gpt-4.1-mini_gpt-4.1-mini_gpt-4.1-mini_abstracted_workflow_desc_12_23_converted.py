async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Clarify and enforce the leading-digit constraints for the two 3-digit numbers formed by the rows. "
        "Explicitly establish that the leading digits (cells a and d) must be nonzero to satisfy the definition of a three-digit number. "
        "Review the problem statement and example to justify this interpretation. Avoid allowing leading zeros in these positions to prevent overcounting invalid solutions. "
        "Document this assumption clearly for all subsequent subtasks."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Stage 0 - {subtask_desc_0['subtask_id']} - Instruction: {cot_instruction_0}")
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, clarifying leading-digit constraints, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    cot_instruction_1_1 = (
        "Sub-task 1: Formulate the digit-wise addition equations for the sum of the two 3-digit row numbers equaling 999, "
        "explicitly incorporating carry-in and carry-out variables for each digit column. Define variables for each digit in the grid (a,b,c,d,e,f) and carry variables (c0, c1, c2) for the units, tens, and hundreds places respectively. "
        "Express the addition as: a + d + c0 = 9 + 10*c1, b + e + c1 = 9 + 10*c2, c + f + c2 = 9 + 10*c3 (where c3=0 since sum is 999). "
        "Emphasize the necessity of considering all possible carry scenarios (0 or 1) to avoid missing valid solutions. Avoid simplifying assumptions that ignore carry effects."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "CoT"
    }
    print(f"Stage 1 - {subtask_desc_1_1['subtask_id']} - Instruction: {cot_instruction_1_1}")
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0, answer_0], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, formulating addition with carry, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_instruction_1_2 = (
        "Sub-task 2: Formulate the column sum constraint: the sum of the three 2-digit numbers formed by reading columns top to bottom equals 99. "
        "Express this as (10*a + d) + (10*b + e) + (10*c + f) = 99. Combine this with the digit variables and carry variables from subtask_1 to establish a system of simultaneous equations and inequalities. "
        "Analyze the implications of these combined constraints on the possible digit values and carry variables. Avoid treating the two sum constraints independently; instead, integrate them to reduce the solution space effectively."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Stage 1 - {subtask_desc_1_2['subtask_id']} - Instruction: {cot_instruction_1_2}")
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, integrating column sum constraint, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Given all the above thinking and answers, synthesize the combined digit and carry constraints." 
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1.2: Synthesize combined constraints." + final_instr_1_2, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_2.id}, synthesizing combined constraints, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    cot_instruction_1_3 = (
        "Sub-task 3: Derive digit range restrictions and relationships from the combined sum and carry equations. "
        "For example, deduce bounds on digits a and d due to the leading-digit nonzero constraint and carry propagation. "
        "Identify feasible carry-in and carry-out combinations for each digit column. Use logical deductions to prune impossible digit assignments early. "
        "Avoid premature enumeration without these constraints to improve efficiency and correctness."
    )
    N_sc_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Stage 1 - {subtask_desc_1_3['subtask_id']} - Instruction: {cot_instruction_1_3}")
    for i in range(N_sc_1_3):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_2, answer_1_2], cot_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, deriving digit range restrictions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_3 = "Given all the above thinking and answers, finalize digit range restrictions and carry feasibility." 
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_answers_1_3 + possible_thinkings_1_3, "Sub-task 1.3: Finalize digit range and carry constraints." + final_instr_1_3, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_1_3.id}, finalizing digit ranges, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Implement a systematic enumeration or constraint satisfaction procedure to generate all candidate digit assignments (a,b,c,d,e,f) that satisfy the digit range constraints, leading-digit nonzero conditions, and carry variable assignments consistent with the addition equations. "
        "For each candidate, verify both sum conditions (row sums with carry and column sums) hold exactly. "
        "Emphasize exhaustive coverage of the solution space without double counting. Avoid ignoring any carry or digit constraints established previously."
    )
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Stage 2 - {subtask_desc_2_1['subtask_id']} - Instruction: {cot_instruction_2_1}")
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_3, answer_1_3], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, enumerating candidate digit assignments, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_1 = "Given all the above thinking and answers, synthesize and confirm the valid digit assignments satisfying both sum conditions." 
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 2.1: Synthesize and confirm valid digit assignments." + final_instr_2_1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_1.id}, synthesizing valid digit assignments, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 2: Filter the enumerated candidate solutions to remove duplicates and confirm the uniqueness of each valid digit assignment. "
        "Verify that each solution respects the problem constraints, including leading-digit nonzero and sum conditions. "
        "Document the count of valid solutions and prepare the data for final aggregation. Avoid counting invalid or repeated solutions."
    )
    N_sc_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Stage 2 - {subtask_desc_2_2['subtask_id']} - Instruction: {cot_instruction_2_2}")
    for i in range(N_sc_2_2):
        thinking_i, answer_i = await cot_agents_2_2[i]([taskInfo, thinking_2_1, answer_2_1], cot_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, filtering and verifying uniqueness, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_2.append(answer_i)
        possible_thinkings_2_2.append(thinking_i)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2_2 = "Given all the above thinking and answers, finalize the unique valid digit assignments count." 
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2.2: Finalize unique valid digit assignments." + final_instr_2_2, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2_2.id}, finalizing unique valid assignments, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Aggregate the filtered valid solutions to compute the total number of distinct digit placements in the 2x3 grid satisfying both sum conditions. "
        "Cross-validate the final count by reapplying the sum constraints to a representative sample of solutions. "
        "Include sanity checks against the example grid and any edge cases identified during enumeration. Avoid errors in aggregation or overlooking any valid solutions."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    print(f"Stage 3 - {subtask_desc_3_1['subtask_id']} - Instruction: {debate_instruction_3_1}")
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking_i, answer_i = await agent([taskInfo, thinking_2_2, answer_2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking_2_2, answer_2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking_i, answer_i = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, aggregating and verifying count, thinking: {thinking_i.content}; answer: {answer_i.content}")
            all_thinking_3_1[r].append(thinking_i)
            all_answer_3_1[r].append(answer_i)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3_1 = "Given all the above thinking and answers, reason carefully and provide the final count of valid digit placements." 
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 3.1: Finalize and verify the total count." + final_instr_3_1, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_3_1.id}, finalizing count, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    reflexion_instruction_3_2 = (
        "Sub-task 2: Reflect on the entire reasoning and counting process. Confirm no valid solutions were missed, no duplicates counted, and the final count aligns with problem constraints. "
        "Provide a final verified number of valid digit placements."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflexion_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    print(f"Stage 3 - {subtask_desc_3_2['subtask_id']} - Instruction: {reflexion_instruction_3_2}")
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflexion_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, reflecting on final count, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    critic_inst_3_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    for i in range(N_max_3_2):
        feedback_3_2, correct_3_2 = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2], "Please review and provide the limitations of provided solutions." + critic_inst_3_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, providing feedback, thinking: {feedback_3_2.content}; answer: {correct_3_2.content}")
        if correct_3_2.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback_3_2])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflexion_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining final count, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
