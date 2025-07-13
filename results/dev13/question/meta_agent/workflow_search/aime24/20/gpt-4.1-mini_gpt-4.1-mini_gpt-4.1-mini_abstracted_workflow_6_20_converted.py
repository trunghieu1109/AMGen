async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Mathematical Formulation and Constraints

    # Sub-task 1: Formulate the key equation (x + y)^2 = x*b + y
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Formulate the mathematical condition for a positive integer n to be b-eautiful by expressing n as n = x*b + y, "
        "where x and y are digits in base b, and derive the key equation (x + y)^2 = x*b + y that must hold. "
        "Emphasize that this equation encapsulates the b-eautiful condition and is the foundation for further analysis."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, formulating key equation, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_answers_0_1 + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent key equation for b-eautiful numbers.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Specify digit constraints for x and y
    cot_instruction_0_2 = (
        "Sub-task 2: Specify the digit constraints for the two-digit number n in base b: the leading digit x must satisfy 1 ≤ x ≤ b-1, "
        "and the second digit y must satisfy 0 ≤ y ≤ b-1. Clarify that these constraints ensure n has exactly two digits in base b and exclude leading zeros."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1.content, answer_0_1.content], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, specifying digit constraints, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Explain why n must be a perfect square and implications
    cot_reflect_instruction_0_3 = (
        "Sub-task 3: Explain why n must be a perfect square and why the sum of digits x + y equals the integer square root of n. "
        "Clarify the implications of this for the search for solutions and the necessity of integer solutions to the derived equation. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_0_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_reflect_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_0_3 = [taskInfo, thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content]
    thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, explaining perfect square condition, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_0_3([taskInfo, thinking_0_3.content, answer_0_3.content], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_0_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_0_3.extend([thinking_0_3.content, answer_0_3.content, feedback.content])
        thinking_0_3, answer_0_3 = await cot_agent_0_3(cot_inputs_0_3, cot_reflect_instruction_0_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_0_3.id}, refining explanation, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Enumeration and Counting of b-eautiful numbers for each base

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Implement a systematic enumeration procedure that, for each base b from 2 to 100, iterates over all valid digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "checks which pairs satisfy the equation (x + y)^2 = x*b + y, and identifies all b-eautiful numbers for that base. "
        "Ensure the enumeration is explicit, exhaustive, and data-driven, avoiding assumptions or hard-coded bases."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_3.content, answer_0_3.content], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, enumerating b-eautiful numbers, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent enumeration results for b-eautiful numbers per base.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 2: Count the total number of b-eautiful numbers found for each base
    cot_instruction_1_2 = (
        "Sub-task 2: Count the total number of b-eautiful numbers found for each base b from the enumeration results. "
        "Ensure the count is accurate, consistent with digit constraints and the key equation, and is stored in a structured format for downstream use."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1.content, answer_1_1.content], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, counting b-eautiful numbers, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Identify minimal base and verify

    # Sub-task 1: Iterate over bases and identify minimal base with count > 10
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Iterate over integer bases b starting from 2 upwards (up to 100), applying the enumeration and counting results. "
        "Collect and record the counts of b-eautiful numbers per base, and identify the minimal base b for which the count exceeds ten. "
        "This subtask must be fully data-driven and avoid guesswork. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_2_1 = [taskInfo, thinking_1_2.content, answer_1_2.content]
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, identifying minimal base, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1.content, answer_2_1.content], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1.content, answer_2_1.content, feedback.content])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining minimal base identification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 2: Verify minimal base by cross-checking smaller bases
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Verify and confirm the minimal base b found in the previous subtask by cross-checking that no smaller base satisfies the condition of having more than ten b-eautiful numbers. "
        "Document the reasoning, enumeration data, and final answer with rigorous justification, ensuring no contradictions or unsupported conclusions."
    )
    cot_sc_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_sc_agents_2_2[i]([taskInfo, thinking_2_1.content, answer_2_1.content, thinking_1_2.content, answer_1_2.content], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_2[i].id}, verifying minimal base, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_answers_2_2 + possible_thinkings_2_2, "Sub-task 2: Synthesize and confirm minimal base verification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 3: Debate phase to reconcile any conflicting conclusions
    debate_instruction_2_3 = (
        "Sub-task 3: Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer to reconcile any conflicting counts or conclusions from the enumeration and verification subtasks. "
        "Ensure synchronization, consistency, and agreement on the final minimal base b, improving the reliability of the overall workflow."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_3 = self.max_round
    all_thinking_2_3 = [[] for _ in range(N_max_2_3)]
    all_answer_2_3 = [[] for _ in range(N_max_2_3)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2_subtask_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_3):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content], debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_1.content, answer_2_1.content, thinking_2_2.content, answer_2_2.content] + all_thinking_2_3[r-1] + all_answer_2_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reconciling minimal base, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_3[r].append(thinking)
            all_answer_2_3[r].append(answer)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1] + all_answer_2_3[-1], "Sub-task 3: Provide final reconciled minimal base answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing minimal base, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_3, answer_2_3, sub_tasks, agents)
    return final_answer, logs
