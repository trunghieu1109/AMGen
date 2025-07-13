async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formal Problem Definition
    cot_instruction_0 = (
        "Sub-task 1: Formally define the problem by expressing any two-digit number n in base b as n = x*b + y, "
        "with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. Define s = x + y and state the key condition s² = n. "
        "Explicitly emphasize that n must satisfy the two-digit number range constraint: b ≤ n < b², ensuring the number has exactly two digits in base b. "
        "Clarify that s must be an integer and that n must be a perfect square. Avoid ambiguity about digit values and base representation rules. "
        "This formalization will serve as the foundation for all subsequent analysis and enumeration.")
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, formal problem definition, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)

    # Stage 1: Derive Key Formula and Constraints
    reflexion_instruction_1 = (
        "Sub-task 1: Analyze and manipulate the key equation s² = x*b + y with s = x + y to derive relationships that isolate variables and establish constraints on s, x, and y. "
        "Specifically, derive expressions for x and y in terms of s and b, such as x = (s² - s)/(b - 1), and identify necessary conditions for x and y to be integers within digit bounds. "
        "Deduce inequalities that limit the range of s given the base b and digit constraints. Emphasize the importance of integer divisibility and digit bounds. "
        "Prepare clear criteria for candidate sums s to be used in enumeration, ensuring these criteria align with the two-digit number constraint and digit bounds.")
    reflexion_agent_1 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflexion_instruction_1,
        "context": ["user query", thinking_0.content, answer_0.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1, answer_1 = await reflexion_agent_1([taskInfo, thinking_0, answer_0], reflexion_instruction_1, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_1.id}, derive formula and constraints, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)

    # Stage 1 Sub-task 2: Define explicit filtering rules for candidate sums s
    reflexion_instruction_1_2 = (
        "Sub-task 2: Establish explicit filtering rules for candidate sums s based on the two-digit number range constraint: "
        "only consider s such that s² satisfies b ≤ s² < b². This ensures that the corresponding n = s² is a two-digit number in base b. "
        "Also, enforce that after computing x = (s² - s)/(b - 1), both x and y = s - x satisfy digit bounds (1 ≤ x ≤ b-1, 0 ≤ y ≤ b-1) and are integers. "
        "Define these conditions formally to guide enumeration in the next stage. This step prevents overcounting and invalid candidates.")
    reflexion_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": reflexion_instruction_1_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "Reflexion | CoT"
    }
    thinking_1_2, answer_1_2 = await reflexion_agent_1_2([taskInfo, thinking_1, answer_1], reflexion_instruction_1_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_1_2.id}, define filtering rules for s, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    # Stage 2 Sub-task 1: Enumeration procedure for fixed base b
    cot_sc_instruction_2_1 = (
        "Sub-task 1: Implement the enumeration procedure for a fixed base b ≥ 2. Enumerate all candidate sums s in the filtered range determined by stage_1.subtask_2. "
        "For each s, compute x = (s² - s)/(b - 1) and y = s - x. Verify that x and y are integers within digit bounds (1 ≤ x ≤ b-1, 0 ≤ y ≤ b-1). "
        "Confirm that n = s² lies within the two-digit number range [b, b²). Collect and store all valid (x,y,n,s) tuples representing b-eautiful numbers. "
        "Log sample valid numbers and their base-b digit representations for transparency. This subtask must strictly enforce all constraints to avoid overcounting.")
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_2_1, answer_2_1 = await cot_sc_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, enumerate b-eautiful numbers, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
        possible_answers_2_1.append(answer_2_1)
        possible_thinkings_2_1.append(thinking_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize enumeration results and enforce constraints strictly.", is_sub_task=True)
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    # Stage 2 Sub-task 2: Brute-force verification for small bases
    reflexion_instruction_2_2 = (
        "Sub-task 2: Perform a brute-force verification for small bases (e.g., b=3,4,5) by enumerating all two-digit numbers (n = x*b + y with digit bounds) and checking the b-eautiful condition directly (s = x + y, s² = n). "
        "Compare counts and valid pairs with results from subtask_1 to validate correctness of the formula-based enumeration method. "
        "Highlight any discrepancies and ensure the two-digit number range constraint is strictly applied in both methods. "
        "This step builds confidence in the enumeration approach and helps detect any logical or implementation errors.")
    reflexion_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Reflexion Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": reflexion_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    thinking_2_2, answer_2_2 = await reflexion_agent_2_2([taskInfo, thinking_2_1, answer_2_1], reflexion_instruction_2_2, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_2_2.id}, brute-force verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    # Stage 2 Sub-task 3: Count b-eautiful numbers for bases starting from 2
    cot_instruction_2_3 = (
        "Sub-task 3: For each base b starting from 2 upwards, use the validated enumeration method to count the number of b-eautiful numbers. "
        "Track these counts and identify the smallest base b for which the count exceeds ten. Implement early stopping upon meeting this condition. "
        "Log counts for intermediate bases to provide insight into growth trends. Ensure no invalid numbers are counted due to improper constraint handling. "
        "Provide a detailed report of counts and candidate numbers for the identified base.")
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_2_1, answer_2_1, thinking_2_2, answer_2_2], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_3.id}, count b-eautiful numbers and find minimal base, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Stage 2 Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)

    # Stage 3 Sub-task 1: Finalize minimal base and verify
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Select and finalize the least integer base b ≥ 2 for which more than ten b-eautiful numbers exist based on stage_2.subtask_3 results. "
        "Re-verify the digit pairs and their sums for this base to ensure no violations of digit bounds or the two-digit number range constraint. "
        "Provide a comprehensive summary of the verification process, including example b-eautiful numbers and their base-b representations. "
        "Deliver the final answer with supporting evidence to demonstrate correctness and robustness of the solution.")
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_3.content, answer_2_3.content],
        "agent_collaboration": "SC_CoT | CoT"
    }
    for i in range(self.max_sc):
        thinking_3_1, answer_3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_3, answer_2_3], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, finalize minimal base and verify, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 1: Synthesize and finalize minimal base with verification.", is_sub_task=True)
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    # Stage 3 Sub-task 2: Reflexion and critique round
    debate_instruction_3_2 = (
        "Sub-task 2: Conduct a reflexion and critique round involving multiple agents to review the entire solution workflow, focusing on constraint enforcement, enumeration correctness, and final answer validation. "
        "Address any lingering doubts or inconsistencies. Confirm that the final solution adheres strictly to problem requirements and that all prior feedback has been incorporated. "
        "Produce a final report documenting the reasoning, verification steps, and conclusion.")
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Debate | Reflexion"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking_3_2, answer_3_2 = await agent([taskInfo, thinking_3_1, answer_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos_3_2 = [taskInfo, thinking_3_1, answer_3_1] + all_thinking_3_2[r-1] + all_answer_3_2[r-1]
                thinking_3_2, answer_3_2 = await agent(input_infos_3_2, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reviewing solution, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
            all_thinking_3_2[r].append(thinking_3_2)
            all_answer_3_2[r].append(answer_3_2)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1] + all_answer_3_2[-1], "Sub-task 2: Final reflexion and critique synthesis.", is_sub_task=True)
    agents.append(f"Final Decision agent, final reflexion and critique, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Stage 3 Sub-task 2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
