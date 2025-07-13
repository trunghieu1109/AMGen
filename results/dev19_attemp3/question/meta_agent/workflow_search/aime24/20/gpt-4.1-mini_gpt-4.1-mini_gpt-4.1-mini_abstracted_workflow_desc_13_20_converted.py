async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Formally define b-eautiful numbers by expressing n in terms of digits d1, d0 and base b, "
        "and derive the key equation n = (d1 + d0)^2 with digit constraints (1 ≤ d1 ≤ b-1, 0 ≤ d0 ≤ b-1, b ≥ 2). "
        "Ensure clarity and correctness to avoid misinterpretation in later stages.")
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, defining b-eautiful numbers, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    reflect_instruction_1 = (
        "Sub-task 1: Combine the digit sum and base representation constraints to derive a composite Diophantine equation "
        "relating digits d1, d0, base b, and the perfect square condition. Simplify and parameterize the problem to characterize valid digit pairs and bases. "
        "Explicitly document all derived formulas and constraints to support enumeration. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking_0, answer_0]
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": reflect_instruction_1,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, deriving composite Diophantine equation, thinking: {thinking_1.content}; answer: {answer_1.content}")
    for i in range(N_max_1):
        feedback_1, correct_1 = await critic_agent_1([taskInfo, thinking_1, answer_1],
                                                   "Please review and provide the limitations of provided solutions. "
                                                   "If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                   i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback_1.content}; correct: {correct_1.content}")
        if correct_1.content.strip() == "True":
            break
        cot_inputs_1.extend([thinking_1, answer_1, feedback_1])
        thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining composite equation, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the parameter ranges and conditions for valid solutions (d1, d0, b) based on the composite equation. "
        "Identify feasible bounds for digit sums and bases to limit the search space for enumeration. "
        "This subtask should prepare precise iteration limits and filtering criteria for the enumeration phase.")
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing parameter ranges, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_1, answer_1] + possible_thinkings_2 + possible_answers_2,
                                                      "Sub-task 2: Synthesize and choose the most consistent and correct parameter ranges and iteration bounds for enumeration.",
                                                      is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 3.1: Perform exhaustive enumeration for each base b from 2 up to a reasonable upper bound (e.g., 30 or higher if needed). "
        "For each b, iterate over all digit pairs (d1, d0) with 1 ≤ d1 ≤ b-1 and 0 ≤ d0 ≤ b-1, check the condition (d1 + d0)^2 = d1 * b + d0 strictly, "
        "and record all valid b-eautiful numbers. This subtask must avoid heuristic or partial checks and produce a complete dataset of counts per base.")
    N_sc_3_1 = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3_1)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3_1):
        thinking_3_1, answer_3_1 = await cot_agents_3_1[i]([taskInfo, thinking_2, answer_2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, enumerating b-eautiful numbers, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo, thinking_2, answer_2] + possible_thinkings_3_1 + possible_answers_3_1,
                                                              "Sub-task 3.1: Synthesize and finalize the enumeration results for b-eautiful numbers per base.",
                                                              is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    reflect_instruction_3_2 = (
        "Sub-task 3.2: Validate and cross-check the enumeration results to ensure all digit and base constraints are strictly satisfied for each candidate. "
        "Count the number of b-eautiful numbers per base and identify the minimal base b ≥ 2 for which the count exceeds ten. "
        "This subtask should include iterative refinement and error checking to prevent premature or incorrect conclusions, leveraging Reflexion to reconcile any discrepancies. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better.")
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_3_1, answer_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflect_instruction_3_2,
        "context": ["user query", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, validating enumeration results, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(N_max_3_2):
        feedback_3_2, correct_3_2 = await critic_agent_3_2([taskInfo, thinking_3_2, answer_3_2],
                                                           "Please review and provide the limitations of provided solutions. "
                                                           "If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                           i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, feedback: {feedback_3_2.content}; correct: {correct_3_2.content}")
        if correct_3_2.content.strip() == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, answer_3_2, feedback_3_2])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining validation, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
