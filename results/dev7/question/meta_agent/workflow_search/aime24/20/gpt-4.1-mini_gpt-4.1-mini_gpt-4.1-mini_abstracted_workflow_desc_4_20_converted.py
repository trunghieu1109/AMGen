async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 0_1: Formally define the problem and establish the mathematical framework. "
        "Express any two-digit number n in base b as n = x*b + y with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "ensuring the leading digit x is nonzero to guarantee exactly two digits. Formulate the key equation (x + y)^2 = x*b + y that characterizes b-eautiful numbers. "
        "Explicitly state and validate all assumptions, including the integer nature of sqrt(n) and the digit bounds. "
        "Clarify that the two-digit condition implies n < b^2 and that this must be enforced in all subsequent steps. "
        "This subtask sets the foundation for all further reasoning and enumeration."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formal problem definition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)

    cot_sc_instruction_1_1 = (
        "Sub-task 1_1: For a given base b, enumerate all digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1. "
        "For each candidate, compute n = x*b + y and explicitly verify the two-digit condition n < b^2. "
        "Check if (x + y)^2 = n holds. Collect and output a structured list of all valid b-eautiful numbers for that base, "
        "including their digit pairs and numeric values. This enumeration must be exhaustive and precise, avoiding off-by-one errors and ensuring no invalid candidates are included. "
        "Output the results as a JSON or table format to serve as a single source of truth for subsequent subtasks."
    )
    N_sc = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", f"thinking: {thinking_0_1.content}", f"answer: {answer_0_1.content}"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating b-eautiful numbers for base b, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1_1: Synthesize consistent enumeration of b-eautiful numbers for given base b.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)

    cot_sc_instruction_1_2 = (
        "Sub-task 1_2: Perform an incremental search over bases b starting from 2 upwards. "
        "For each base, invoke subtask_1_1 to enumerate all b-eautiful numbers and count them. "
        "Maintain a structured record of counts and candidate lists per base. Stop the search as soon as a base is found with more than ten b-eautiful numbers. "
        "Manage computations efficiently, store results systematically, and produce a summary table of counts per base. "
        "Pass enumeration results forward_20 for validation and final decision-making."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_1_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", f"thinking: {thinking_1_1.content}", f"answer: {answer_1_1.content}"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, counting b-eautiful numbers over bases, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 1_2: Synthesize consistent count and identify minimal base b.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)

    cot_reflect_instruction_1_3 = (
        "Sub-task 1_3: Validate enumeration results from subtasks 1_1 and 1_2 by explicitly checking that all candidate numbers satisfy the two-digit condition (n < b^2) and digit bounds. "
        "Identify and discard any candidates violating these constraints. Confirm that the counts of b-eautiful numbers per base are accurate and free from overcounting. "
        "This subtask acts as a quality control step to ensure the integrity of the enumeration data before it is used in further analysis."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_3 = self.max_round
    cot_inputs_1_3 = [taskInfo, thinking_1_2, answer_1_2, thinking_1_1, answer_1_1, thinking_0_1, answer_0_1]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_1_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", f"thinking: {thinking_1_2.content}", f"answer: {answer_1_2.content}", f"thinking: {thinking_1_1.content}", f"answer: {answer_1_1.content}"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, validating enumeration, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(N_max_1_3):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide limitations of provided enumeration validation. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, feedback: {feedback_1_3.content}; correct: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await cot_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_3.id}, refining validation after feedback, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)

    debate_instruction_2_1 = (
        "Sub-task 2_1: Analyze the key equation (x + y)^2 = x*b + y to derive algebraic simplifications, patterns, or bounds that can reduce computational complexity or provide insight into the distribution of solutions. "
        "For example, express y in terms of x and b or vice versa, derive inequalities, or identify digit sum constraints. "
        "Use these insights to cross-validate the enumeration results and to understand structural properties of b-eautiful numbers. "
        "This subtask supports optimization and verification of the enumeration process."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_2_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", f"thinking: {thinking_1_1.content}", f"answer: {answer_1_1.content}"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing equation patterns, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 2_1: Synthesize debate insights on equation simplification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)

    cot_reflect_instruction_2_2 = (
        "Sub-task 2_2: Apply the algebraic insights from subtask_2_1 to optimize or verify the enumeration process. "
        "Check for missed solutions or extraneous candidates in the enumeration data. Confirm that the enumeration and counting are consistent with the theoretical analysis. "
        "This subtask ensures the reliability and correctness of the search results and prepares validated data for final aggregation."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_3, answer_1_3]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", f"thinking: {thinking_2_1.content}", f"answer: {answer_2_1.content}", f"thinking: {thinking_1_3.content}", f"answer: {answer_1_3.content}"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining enumeration verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review and provide limitations of provided verification. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback: {feedback_2_2.content}; correct: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining after feedback, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)

    cot_instruction_3_1 = (
        "Sub-task 3_1: Aggregate the validated counts of b-eautiful numbers for each base b from the enumeration and analysis subtasks. "
        "Identify the smallest base b ≥ 2 for which the count exceeds ten. Present the final answer clearly, supported by the enumeration data and algebraic verification. "
        "Provide a concise explanation or proof of correctness, referencing the validation steps to confirm no errors or overcounting occurred. "
        "Return the minimal base alongside the verification results to conclude the problem-solving process."
    )
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_3_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", f"thinking: {thinking_1_3.content}", f"answer: {answer_1_3.content}", f"thinking: {thinking_2_2.content}", f"answer: {answer_2_2.content}"],
        "agent_collaboration": "CoT"
    }
    for i in range(N_sc):
        thinking_3_1, answer_3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking_1_3, answer_1_3, thinking_2_2, answer_2_2], cot_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, aggregating counts and identifying minimal base, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 3_1: Synthesize final answer for minimal base b.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
