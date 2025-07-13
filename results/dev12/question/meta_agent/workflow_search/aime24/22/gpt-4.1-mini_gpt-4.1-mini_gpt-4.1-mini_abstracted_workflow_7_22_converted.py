async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Initial Formalization and Constraint Analysis

    # Sub-task 1: Define list variables and sum constraint (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally define the list variables: specify the list length (n), the elements as positive integers, "
        "and their frequencies. Express the sum constraint as sum of all elements equals 30. Emphasize positivity and unknown length."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, defining list variables and sum constraint, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    # Sub-task 2: Analyze median condition (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Analyze the median condition independently: deduce that the list length must be even because the median is a positive integer not present in the list. "
        "Formalize that the median equals the average of the two middle elements in the sorted list, which must be an integer not in the list."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, analyzing median condition, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)

    # Sub-task 3: Analyze mode condition (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Analyze the mode condition independently: define the unique mode as the integer 9 appearing strictly more times than any other element in the list. "
        "Formalize the strict inequality for frequencies: frequency(9) > frequency(x) for all x ≠ 9. Emphasize frequency(9) ≥ 2 and no ties allowed."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, analyzing mode condition, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)

    # Sub-task 4: Combine constraints (Self-Consistency CoT)
    cot_sc_instruction_0_4 = (
        "Sub-task 4: Combine the constraints from sum, median, and mode analyses to derive initial restrictions on the list length, element values, and frequency distributions. "
        "Summarize implications: list length even, median integer not in list, 9 unique mode with strictly highest frequency, sum equals 30. Avoid premature enumeration."
    )
    N_sc_0_4 = self.max_sc
    cot_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_4)]
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_sc_instruction_0_4,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_4):
        thinking_i, answer_i = await cot_agents_0_4[i]([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_4[i].id}, combining constraints, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_4.append(answer_i)
        possible_thinkings_0_4.append(thinking_i)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + possible_answers_0_4 + possible_thinkings_0_4, "Sub-task 4: Synthesize and choose the most consistent combined constraints summary.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing combined constraints, thinking: {thinking_0_4.content}; answer: {answer_0_4.content}")
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)

    # Stage 1: Enumerate lengths, median pairs, frequency distributions, candidate lists, and validate

    # Sub-task 1: Enumerate possible even list lengths and median pairs (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible even list lengths (n) consistent with sum and median constraints. "
        "For each n, enumerate all possible pairs of middle elements (m1, m2) such that their average is a positive integer median not present in the list. "
        "Explicitly list median pairs and median values. Avoid assuming median values without verification."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_4, answer_0_4], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerating list lengths and median pairs, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    # Sub-task 2: Generate all possible frequency distributions with strict unique mode (Self-Consistency CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Generate all possible frequency distributions of positive integers including 9 that sum to list length n (from subtask 1). "
        "Enforce frequency(9) strictly greater than all others, frequency(9) ≥ 2. Filter out any distributions violating unique mode. "
        "Ensure sum of elements weighted by frequencies equals 30. Avoid frequency ties or violations."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_3.content, answer_0_3.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking_i, answer_i = await cot_agents_1_2[i]([taskInfo, thinking_0_3, answer_0_3, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, generating frequency distributions, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_2.append(answer_i)
        possible_thinkings_1_2.append(thinking_i)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose consistent frequency distributions with strict unique mode.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing frequency distributions, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Sub-task 3: Enumerate candidate lists consistent with frequencies and median (CoT)
    cot_instruction_1_3 = (
        "Sub-task 3: For each valid frequency distribution from subtask 2, enumerate candidate lists of positive integers consistent with frequencies and sum constraints. "
        "Verify median matches average of two middle elements and median not in list. Check unique mode again. Reject invalid candidates."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content, thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2, answer_1_2, thinking_1_1, answer_1_1], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, enumerating candidate lists, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    # Sub-task 4: Validate candidate lists explicitly (Self-Consistency CoT)
    cot_sc_instruction_1_4 = (
        "Sub-task 4: Implement explicit validation for candidate lists from subtask 3: confirm sum=30, median positive integer not in list, and 9 unique mode strictly highest frequency. "
        "Reject any candidate failing criteria before sum-of-squares calculation."
    )
    N_sc_1_4 = self.max_sc
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_4)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_4):
        thinking_i, answer_i = await cot_agents_1_4[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, validating candidate lists, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_4.append(answer_i)
        possible_thinkings_1_4.append(thinking_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_answers_1_4 + possible_thinkings_1_4, "Sub-task 4: Synthesize and choose valid candidate lists passing all constraints.", is_sub_task=True)
    agents.append(f"Final Decision agent, validating candidate lists, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)

    # Stage 2: Compute sum of squares and finalize answer

    # Sub-task 1: Compute sum of squares for each valid candidate list (CoT)
    cot_instruction_2_1 = (
        "Sub-task 1: For each candidate list passing validation in stage_1.subtask_4, compute the sum of the squares of all items. "
        "Document calculation process and results clearly."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_4, answer_1_4], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing sum of squares, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Sub-task 2: Analyze sums of squares and finalize solution with reflexion (Reflexion)
    reflect_inst_2_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Analyze the computed sums of squares from subtask 1 to identify the unique valid solution. "
        "Confirm solution satisfies all problem constraints and finalize the answer. "
        "Include reflexion step to critique the solution, ensuring no overlooked violations, especially unique mode condition. "
        "If violations found, prompt reconsideration of candidate lists. " + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, analyzing sums of squares and finalizing, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of provided solutions. If absolutely correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining solution, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
